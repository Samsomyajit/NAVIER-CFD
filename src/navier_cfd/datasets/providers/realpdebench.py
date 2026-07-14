from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core import CFDSample, DatasetAdapterError
from ..huggingface import HuggingFaceDatasetManager
from .common import (
    ScientificDatasetAccessPlan,
    coordinate_grid,
    flatten_time_channels,
    resolve_cache_dir,
    split_members,
)


REALPDEBENCH_REPO = "AI4Science-WestlakeU/RealPDEBench"
REALPDEBENCH_SCENARIOS = {
    "cylinder",
    "controlled_cylinder",
    "fsi",
    "foil",
    "combustion",
}


def _decode_array(value: Any, shape: Sequence[int], *, dtype: Any = np.float32) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, memoryview)):
        array = np.frombuffer(value, dtype=dtype)
    else:
        array = np.asarray(value, dtype=dtype)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise DatasetAdapterError(
            f"RealPDEBench field contains {array.size} values but shape {tuple(shape)} requires {expected}"
        )
    return array.reshape(tuple(int(item) for item in shape))


def _fluid_fields(
    row: Mapping[str, Any],
    data_type: str,
    fields: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
    requested = tuple(fields or (("u", "v", "p") if data_type == "numerical" else ("u", "v")))
    arrays = []
    names = []
    for field in requested:
        if field not in row or row[field] is None:
            continue
        arrays.append(_decode_array(row[field], shape))
        names.append(field)
    if not arrays:
        raise DatasetAdapterError(f"No requested fluid fields {requested} were present in RealPDEBench row")
    return np.stack(arrays, axis=-1), tuple(names)


def _combustion_fields(
    row: Mapping[str, Any],
    data_type: str,
    fields: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
    if data_type == "real":
        key = "observed"
        if key not in row:
            raise DatasetAdapterError("Real combustion row has no observed field")
        return _decode_array(row[key], shape)[..., None], (key,)
    channels = int(row.get("numerical_channels", 15))
    full = _decode_array(row["numerical"], (*shape, channels))
    if fields:
        indices = [int(field) for field in fields]
        return full[..., indices], tuple(f"numerical_{index}" for index in indices)
    return full, tuple(f"numerical_{index}" for index in range(channels))


def _coordinates(row: Mapping[str, Any]) -> np.ndarray | None:
    height = int(row["shape_h"])
    width = int(row["shape_w"])
    vectors = []
    for key in ("x", "y"):
        if key in row and row[key] is not None:
            vectors.append(_decode_array(row[key], (height, width)))
    return coordinate_grid(vectors)


class RealPDEBenchTrajectoryDataset:
    """Window complete trajectories from selectively downloaded Arrow shards."""

    def __init__(
        self,
        rows: Any,
        *,
        scenario: str,
        data_type: str = "real",
        split: str = "train",
        n_steps_input: int = 20,
        n_steps_output: int = 20,
        time_stride: int = 1,
        window_stride: int = 20,
        fields: Sequence[str] | None = None,
        trajectory_limit: int | None = None,
        max_windows: int | None = None,
        seed: int = 42,
        access_plan: Mapping[str, Any] | None = None,
    ) -> None:
        if scenario not in REALPDEBENCH_SCENARIOS:
            raise ValueError(f"Unknown RealPDEBench scenario {scenario!r}")
        if data_type not in {"real", "numerical"}:
            raise ValueError("data_type must be real or numerical")
        self.rows = rows
        self.scenario = scenario
        self.data_type = data_type
        self.split = split
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.time_stride = time_stride
        self.window_stride = window_stride
        self.fields = tuple(fields or ())
        self.access_plan = dict(access_plan or {})

        trajectory_ids = split_members(len(rows), split, seed=seed)
        if trajectory_limit is not None:
            trajectory_ids = trajectory_ids[:trajectory_limit]
        windows: list[tuple[int, int]] = []
        horizon = (n_steps_input + n_steps_output - 1) * time_stride + 1
        for trajectory_id in trajectory_ids:
            row = rows[trajectory_id]
            time_count = int(row["shape_t"])
            starts = range(0, max(0, time_count - horizon + 1), max(1, window_stride))
            windows.extend((trajectory_id, start) for start in starts)
        if max_windows is not None:
            windows = windows[:max_windows]
        if not windows:
            raise DatasetAdapterError(
                "No RealPDEBench windows remain; reduce the horizon or increase downloaded trajectories"
            )
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> CFDSample:
        trajectory_id, start = self.windows[index]
        row = self.rows[trajectory_id]
        if self.scenario == "combustion":
            trajectory, field_names = _combustion_fields(row, self.data_type, self.fields or None)
        else:
            trajectory, field_names = _fluid_fields(row, self.data_type, self.fields or None)
        time_indices = start + np.arange(self.n_steps_input + self.n_steps_output) * self.time_stride
        selected = trajectory[time_indices]
        inputs = flatten_time_channels(selected[: self.n_steps_input])
        targets = flatten_time_channels(selected[self.n_steps_input :])
        coordinates = _coordinates(row)
        time_grid = None
        if "t" in row and row["t"] is not None:
            time_grid = _decode_array(row["t"], (int(row["shape_t"]),))[time_indices]
        return CFDSample(
            inputs=inputs.astype(np.float32, copy=False),
            targets=targets.astype(np.float32, copy=False),
            coordinates=coordinates,
            metadata={
                "dataset_id": "realpdebench",
                "provider": "realpdebench_arrow",
                "configuration": self.scenario,
                "data_type": self.data_type,
                "source_split": self.split,
                "sim_id": row.get("sim_id", str(trajectory_id)),
                "trajectory_index": trajectory_id,
                "window_start": start,
                "field_names": field_names,
                "input_steps": self.n_steps_input,
                "output_steps": self.n_steps_output,
                "time_stride": self.time_stride,
                "time_grid": time_grid,
                "representation": "structured",
                "layout": "spatial_channel_last_time_flattened",
                "subset_mode": True,
                "official_split": False,
                "access_plan": self.access_plan,
            },
        )


class RealPDEBenchDatasetManager:
    """Selective Arrow-shard loader for RealPDEBench small-compute experiments."""

    def __init__(
        self,
        token: str | bool | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.hub = HuggingFaceDatasetManager(token=token, endpoint=endpoint)

    @staticmethod
    def _datasets() -> tuple[Any, Any]:
        try:
            from datasets import Dataset, concatenate_datasets
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("RealPDEBench Arrow support requires the datasets package") from exc
        return Dataset, concatenate_datasets

    @staticmethod
    def validate_configuration(configuration: str) -> str:
        key = configuration.lower().replace("-", "_").replace(" ", "_")
        if key not in REALPDEBENCH_SCENARIOS:
            raise ValueError(
                f"Unknown RealPDEBench configuration {configuration!r}; choose one of "
                f"{sorted(REALPDEBENCH_SCENARIOS)}"
            )
        return key

    def probe(self, *, revision: str | None = None) -> Any:
        return self.hub.probe(REALPDEBENCH_REPO, revision=revision)

    def _select_arrow_files(
        self,
        scenario: str,
        data_type: str,
        *,
        revision: str | None,
        max_arrow_files: int,
        max_file_size_gb: float | None,
    ) -> list[str]:
        prefix = f"{scenario}/hf_dataset/{data_type}/"
        candidates = []
        for entry in self.hub.list_file_entries(REALPDEBENCH_REPO, revision=revision):
            path = str(entry["path"])
            if not path.startswith(prefix) or not path.lower().endswith(".arrow"):
                continue
            size = entry.get("size")
            if max_file_size_gb is not None and size is not None:
                if float(size) > max_file_size_gb * 1024**3:
                    continue
            candidates.append((float("inf") if size is None else int(size), path))
        if not candidates:
            raise FileNotFoundError(
                "No RealPDEBench Arrow shard matched the selected scenario/type/size limit."
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [path for _, path in candidates[: max(1, max_arrow_files)]]

    def load(
        self,
        configuration: str,
        *,
        split: str = "train",
        data_type: str = "real",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        max_arrow_files: int = 1,
        max_file_size_gb: float | None = 2.5,
        **dataset_kwargs: Any,
    ) -> RealPDEBenchTrajectoryDataset:
        scenario = self.validate_configuration(configuration)
        if data_type not in {"real", "numerical"}:
            raise ValueError("data_type must be real or numerical")
        files = self._select_arrow_files(
            scenario,
            data_type,
            revision=revision,
            max_arrow_files=max_arrow_files,
            max_file_size_gb=max_file_size_gb,
        )
        cache = resolve_cache_dir(cache_dir, "realpdebench")
        local_files = [
            self.hub.download_file(
                REALPDEBENCH_REPO,
                filename,
                revision=revision,
                cache_dir=cache,
            )
            for filename in files
        ]
        Dataset, concatenate_datasets = self._datasets()
        shards = [Dataset.from_file(str(path)) for path in local_files]
        rows = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        resolved_revision = self.hub.resolve_revision(REALPDEBENCH_REPO, revision)
        plan = ScientificDatasetAccessPlan(
            provider="realpdebench_arrow",
            dataset_id="realpdebench",
            configuration=scenario,
            split=split,
            repo_id=REALPDEBENCH_REPO,
            revision=revision,
            resolved_revision=resolved_revision,
            files=tuple(files),
            subset_mode=True,
            official_split=False,
            auth_source=self.hub.auth.source,
            notes=(
                "Small-compute mode loads selected Arrow shards only.",
                "Splits are deterministic by loaded trajectory and are not the official full benchmark split.",
            ),
        )
        return RealPDEBenchTrajectoryDataset(
            rows,
            scenario=scenario,
            data_type=data_type,
            split=split,
            access_plan=plan.to_dict(),
            **dataset_kwargs,
        )


__all__ = [
    "REALPDEBENCH_REPO",
    "REALPDEBENCH_SCENARIOS",
    "RealPDEBenchDatasetManager",
    "RealPDEBenchTrajectoryDataset",
]
