from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..core import CFDSample, DatasetAdapterError
from ..huggingface import HuggingFaceDatasetManager
from .common import (
    ScientificDatasetAccessPlan,
    coordinate_grid,
    ensure_channel_last_time_series,
    flatten_time_channels,
    resolve_cache_dir,
    split_members,
)


PDEBENCH_REPOSITORIES: dict[str, str] = {
    "burgers": "pdebench/Burgers",
    "advection": "pdebench/Advection",
    "compressible_navier_stokes_1d": "pdebench/1D-Compressible-Navier-Stokes",
}


class MissingPDEBenchDependency(RuntimeError):
    pass


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingPDEBenchDependency(
            "PDEBench HDF5 support requires `pip install navier-cfd[pdebench]`."
        ) from exc
    return h5py


def _all_datasets(group: Any, prefix: str = "") -> dict[str, Any]:
    h5py = _h5py()
    result: dict[str, Any] = {}
    for name, value in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            result[path] = value
        elif isinstance(value, h5py.Group):
            result.update(_all_datasets(value, path))
    return result


def _choose_tensor_key(datasets: Mapping[str, Any], requested: str | None = None) -> str:
    if requested:
        if requested not in datasets:
            raise DatasetAdapterError(
                f"PDEBench tensor_key={requested!r} not found; available keys: {sorted(datasets)}"
            )
        return requested
    preferred = ("tensor", "solution", "data", "u", "fields")
    candidates = []
    for key, dataset in datasets.items():
        if getattr(dataset, "ndim", 0) < 3:
            continue
        lower = key.lower()
        if any(token in lower for token in ("coordinate", "grid", "time", "x", "y", "z")):
            if dataset.ndim <= 2:
                continue
        score = 0
        for rank, token in enumerate(preferred):
            if lower == token or lower.endswith(f"/{token}"):
                score += 100 - rank
        score += min(int(np.prod(dataset.shape[2:])), 10_000_000) / 10_000_000
        candidates.append((score, key))
    if not candidates:
        raise DatasetAdapterError("No trajectory-like numeric HDF5 dataset was found")
    return max(candidates)[1]


def _find_coordinate_arrays(datasets: Mapping[str, Any]) -> list[np.ndarray]:
    ordered: list[tuple[int, np.ndarray]] = []
    names = {
        "x-coordinate": 0,
        "x_coordinate": 0,
        "x": 0,
        "y-coordinate": 1,
        "y_coordinate": 1,
        "y": 1,
        "z-coordinate": 2,
        "z_coordinate": 2,
        "z": 2,
    }
    for key, dataset in datasets.items():
        leaf = key.rsplit("/", 1)[-1].lower()
        if leaf in names and dataset.ndim in (1, 2, 3):
            ordered.append((names[leaf], np.asarray(dataset)))
    ordered.sort(key=lambda item: item[0])
    return [array for _, array in ordered]


class PDEBenchHDF5Dataset:
    """Lazy windowed view over one scientific PDEBench HDF5 file."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str = "train",
        configuration: str = "unknown",
        tensor_key: str | None = None,
        sample_axis: int = 0,
        time_axis: int = 1,
        channel_axis: int | None = None,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        time_stride: int = 1,
        window_stride: int = 1,
        trajectory_limit: int | None = None,
        max_windows: int | None = None,
        seed: int = 42,
        split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
        access_plan: Mapping[str, Any] | None = None,
    ) -> None:
        if sample_axis != 0:
            raise ValueError("PDEBenchHDF5Dataset currently requires sample_axis=0")
        if n_steps_input < 1 or n_steps_output < 1:
            raise ValueError("n_steps_input and n_steps_output must be positive")
        self.path = Path(path)
        self.split = split
        self.configuration = configuration
        self.sample_axis = sample_axis
        self.time_axis = time_axis
        self.channel_axis = channel_axis
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.time_stride = time_stride
        self.window_stride = window_stride
        self.access_plan = dict(access_plan or {})

        h5py = _h5py()
        with h5py.File(self.path, "r") as handle:
            datasets = _all_datasets(handle)
            self.tensor_key = _choose_tensor_key(datasets, tensor_key)
            tensor = datasets[self.tensor_key]
            if tensor.ndim < 3:
                raise DatasetAdapterError(
                    f"PDEBench tensor {self.tensor_key!r} must have sample, time, and space axes"
                )
            self.raw_shape = tuple(int(value) for value in tensor.shape)
            sample_count = self.raw_shape[sample_axis]
            time_count = self.raw_shape[time_axis]
            coordinates = _find_coordinate_arrays(datasets)
            self.coordinates = coordinate_grid(coordinates)

        members = split_members(
            sample_count,
            split,
            seed=seed,
            fractions=split_fractions,
        )
        if trajectory_limit is not None:
            members = members[:trajectory_limit]
        horizon = (n_steps_input + n_steps_output - 1) * time_stride + 1
        starts = range(0, max(0, time_count - horizon + 1), max(1, window_stride))
        windows = [(member, start) for member in members for start in starts]
        if max_windows is not None:
            windows = windows[:max_windows]
        if not windows:
            raise DatasetAdapterError(
                "No PDEBench windows remain; reduce the input/output horizon or choose more trajectories"
            )
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> CFDSample:
        h5py = _h5py()
        trajectory_index, start = self.windows[index]
        with h5py.File(self.path, "r") as handle:
            raw = np.asarray(handle[self.tensor_key][trajectory_index])
        time_axis_after_sample = self.time_axis - 1 if self.time_axis > self.sample_axis else self.time_axis
        channel_axis_after_sample = self.channel_axis
        if channel_axis_after_sample is not None and channel_axis_after_sample >= 0:
            channel_axis_after_sample -= 1 if channel_axis_after_sample > self.sample_axis else 0
        trajectory = ensure_channel_last_time_series(
            raw,
            time_axis=time_axis_after_sample,
            channel_axis=channel_axis_after_sample,
        )
        times = start + np.arange(self.n_steps_input + self.n_steps_output) * self.time_stride
        selected = trajectory[times]
        inputs = flatten_time_channels(selected[: self.n_steps_input])
        targets = flatten_time_channels(selected[self.n_steps_input :])
        coordinates = self.coordinates
        if coordinates is None:
            spatial_shape = inputs.shape[:-1]
            axes = [np.linspace(0.0, 1.0, size, dtype=np.float32) for size in spatial_shape]
            coordinates = coordinate_grid(axes)
        metadata = {
            "dataset_id": "pdebench",
            "provider": "pdebench_hdf5",
            "configuration": self.configuration,
            "source_file": self.path.name,
            "tensor_key": self.tensor_key,
            "trajectory_index": trajectory_index,
            "window_start": start,
            "input_steps": self.n_steps_input,
            "output_steps": self.n_steps_output,
            "time_stride": self.time_stride,
            "representation": "structured",
            "layout": "spatial_channel_last_time_flattened",
            "source_split": self.split,
            "access_plan": self.access_plan,
        }
        return CFDSample(
            inputs=inputs.astype(np.float32, copy=False),
            targets=targets.astype(np.float32, copy=False),
            coordinates=coordinates,
            metadata=metadata,
        )


class PDEBenchDatasetManager:
    """Selective Hugging Face HDF5 access for PDEBench repositories."""

    def __init__(
        self,
        token: str | bool | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.hub = HuggingFaceDatasetManager(token=token, endpoint=endpoint)

    @staticmethod
    def repository(configuration: str, repo_id: str | None = None) -> str:
        if repo_id:
            return repo_id
        key = configuration.lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "burgers_1d": "burgers",
            "1d_burgers": "burgers",
            "advection_1d": "advection",
            "1d_advection": "advection",
            "compressible_ns_1d": "compressible_navier_stokes_1d",
            "1d_compressible_navier_stokes": "compressible_navier_stokes_1d",
            "compressible_navier_stokes": "compressible_navier_stokes_1d",
        }
        key = aliases.get(key, key)
        try:
            return PDEBENCH_REPOSITORIES[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown built-in PDEBench configuration {configuration!r}. "
                f"Choose one of {sorted(PDEBENCH_REPOSITORIES)} or pass repo_id explicitly."
            ) from exc

    def _select_file(
        self,
        repo_id: str,
        *,
        revision: str | None,
        filename: str | None,
        file_pattern: str | None,
        max_file_size_gb: float | None,
    ) -> tuple[str, int | None]:
        entries = self.hub.list_file_entries(repo_id, revision=revision)
        candidates = []
        for entry in entries:
            path = str(entry["path"])
            if not path.lower().endswith((".h5", ".hdf5")):
                continue
            if filename and path != filename:
                continue
            if file_pattern and not fnmatch(path, file_pattern):
                continue
            size = entry.get("size")
            if max_file_size_gb is not None and size is not None:
                if float(size) > max_file_size_gb * 1024**3:
                    continue
            candidates.append((float("inf") if size is None else int(size), path, size))
        if not candidates:
            raise FileNotFoundError(
                "No matching PDEBench HDF5 file was found. Supply filename or file_pattern; "
                "use navier datasets probe pdebench --configuration <name> to inspect files."
            )
        _, path, size = min(candidates, key=lambda item: (item[0], item[1]))
        return path, size

    def probe(self, configuration: str, *, repo_id: str | None = None, revision: str | None = None) -> Any:
        return self.hub.probe(self.repository(configuration, repo_id), revision=revision)

    def load(
        self,
        configuration: str,
        *,
        split: str = "train",
        repo_id: str | None = None,
        revision: str | None = None,
        filename: str | None = None,
        file_pattern: str | None = None,
        cache_dir: str | Path | None = None,
        max_file_size_gb: float | None = None,
        **dataset_kwargs: Any,
    ) -> PDEBenchHDF5Dataset:
        resolved_repo = self.repository(configuration, repo_id)
        selected, _ = self._select_file(
            resolved_repo,
            revision=revision,
            filename=filename,
            file_pattern=file_pattern,
            max_file_size_gb=max_file_size_gb,
        )
        cache = resolve_cache_dir(cache_dir, "pdebench")
        local = self.hub.download_file(
            resolved_repo,
            selected,
            revision=revision,
            cache_dir=cache,
        )
        resolved_revision = self.hub.resolve_revision(resolved_repo, revision)
        plan = ScientificDatasetAccessPlan(
            provider="pdebench_hdf5",
            dataset_id="pdebench",
            configuration=configuration,
            split=split,
            repo_id=resolved_repo,
            revision=revision,
            resolved_revision=resolved_revision,
            files=(selected,),
            subset_mode=True,
            official_split=False,
            auth_source=self.hub.auth.source,
            notes=("Trajectory-level deterministic subset split; source HDF5 has no datasets.load_dataset schema.",),
        )
        return PDEBenchHDF5Dataset(
            local,
            split=split,
            configuration=configuration,
            access_plan=plan.to_dict(),
            **dataset_kwargs,
        )


__all__ = [
    "MissingPDEBenchDependency",
    "PDEBENCH_REPOSITORIES",
    "PDEBenchDatasetManager",
    "PDEBenchHDF5Dataset",
]
