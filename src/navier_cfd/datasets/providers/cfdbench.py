from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from ..core import CFDSample, DatasetAdapterError
from ..huggingface import HuggingFaceDatasetManager
from .common import ListCFDDataset, ScientificDatasetAccessPlan, resolve_cache_dir, split_members


CFDBENCH_REPO = "chen-yingfa/CFDBench-raw"
CFDBENCH_SCENARIOS = {
    "cavity": "01_cavityflow",
    "cavityflow": "01_cavityflow",
    "tube": "02_tubeflow",
    "tubeflow": "02_tubeflow",
    "dam": "03_damflow",
    "damflow": "03_damflow",
    "cylinder": "04_cylinderflow",
    "cylinderflow": "04_cylinderflow",
}


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path)))


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise DatasetAdapterError(f"Unsafe archive member: {member.filename}")
        handle.extractall(destination)


def _mapping_to_sample(
    mapping: Mapping[str, Any],
    *,
    source_file: str,
    scenario: str,
) -> CFDSample:
    normalised = {_normalise_name(str(key)): np.asarray(value) for key, value in mapping.items()}
    x = next(
        (
            normalised[key]
            for key in ("x_coordinate", "xcoordinate", "x", "coord_x")
            if key in normalised
        ),
        None,
    )
    y = next(
        (
            normalised[key]
            for key in ("y_coordinate", "ycoordinate", "y", "coord_y")
            if key in normalised
        ),
        None,
    )
    z = next(
        (
            normalised[key]
            for key in ("z_coordinate", "zcoordinate", "z", "coord_z")
            if key in normalised
        ),
        None,
    )
    coordinates = None
    coordinate_values = [value for value in (x, y, z) if value is not None]
    if coordinate_values:
        flat = [np.asarray(value).reshape(-1) for value in coordinate_values]
        if all(len(value) == len(flat[0]) for value in flat):
            coordinates = np.stack(flat, axis=-1).astype(np.float32)

    preferred = (
        "x_velocity",
        "xvelocity",
        "y_velocity",
        "yvelocity",
        "z_velocity",
        "zvelocity",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "u",
        "v",
        "w",
        "absolute_pressure",
        "absolutepressure",
        "pressure",
        "velocity_magnitude",
        "velocitymagnitude",
    )
    target_names = [key for key in preferred if key in normalised]
    if not target_names:
        excluded = {
            "nodenumber",
            "node_number",
            "x_coordinate",
            "xcoordinate",
            "y_coordinate",
            "ycoordinate",
            "z_coordinate",
            "zcoordinate",
            "x",
            "y",
            "z",
            "stream_function",
        }
        target_names = [
            key
            for key, value in normalised.items()
            if key not in excluded and np.issubdtype(np.asarray(value).dtype, np.number)
        ]
    if not target_names:
        raise DatasetAdapterError(f"No CFD target fields were identified in {source_file}")
    targets = np.stack([normalised[key].reshape(-1) for key in target_names], axis=-1).astype(np.float32)
    if coordinates is None:
        coordinates = np.arange(len(targets), dtype=np.float32)[:, None]

    return CFDSample(
        inputs=coordinates.copy(),
        targets=targets,
        coordinates=coordinates,
        metadata={
            "dataset_id": "cfdbench",
            "provider": "cfdbench_archive",
            "configuration": scenario,
            "source_file": source_file,
            "target_fields": tuple(target_names),
            "representation": "point_cloud",
        },
    )


def _load_text(path: Path, scenario: str) -> CFDSample:
    try:
        data = np.genfromtxt(
            path,
            names=True,
            dtype=np.float64,
            encoding="utf-8",
            invalid_raise=False,
        )
        if data.dtype.names:
            mapping = {name: np.asarray(data[name]) for name in data.dtype.names}
            return _mapping_to_sample(mapping, source_file=path.name, scenario=scenario)
    except (ValueError, UnicodeDecodeError):
        pass
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] < 3:
        raise DatasetAdapterError(f"Text field {path} has fewer than three columns")
    mapping: dict[str, Any] = {
        "x_coordinate": matrix[:, 0],
        "y_coordinate": matrix[:, 1],
    }
    for index in range(2, matrix.shape[1]):
        mapping[f"field_{index - 2}"] = matrix[:, index]
    return _mapping_to_sample(mapping, source_file=path.name, scenario=scenario)


def _load_npz(path: Path, scenario: str) -> CFDSample:
    with np.load(path, allow_pickle=False) as data:
        return _mapping_to_sample(
            {key: data[key] for key in data.files},
            source_file=path.name,
            scenario=scenario,
        )


def _load_npy(path: Path, scenario: str) -> CFDSample:
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2 or value.shape[1] < 3:
        raise DatasetAdapterError(
            f"NumPy field {path} must have shape [points, columns>=3]; got {value.shape}"
        )
    mapping: dict[str, Any] = {
        "x_coordinate": value[:, 0],
        "y_coordinate": value[:, 1],
    }
    for index in range(2, value.shape[1]):
        mapping[f"field_{index - 2}"] = value[:, index]
    return _mapping_to_sample(mapping, source_file=path.name, scenario=scenario)


def load_cfdbench_archive_samples(
    archive: str | Path,
    *,
    scenario: str,
    extract_dir: str | Path,
    max_samples: int | None = None,
    sample_stride: int = 1,
) -> list[CFDSample]:
    archive = Path(archive)
    extract_dir = Path(extract_dir)
    marker = extract_dir / ".complete"
    if not marker.exists():
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        _safe_extract(archive, extract_dir)
        marker.write_text("ok", encoding="utf-8")

    supported = {".npz", ".npy", ".csv", ".txt", ".dat"}
    files = sorted(
        (path for path in extract_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported),
        key=_natural_key,
    )
    files = files[:: max(1, sample_stride)]
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise DatasetAdapterError(
            "The selected CFDBench archive contains no supported NPZ/NPY/CSV/TXT/DAT fields. "
            "Pickle files are intentionally not executed for security."
        )

    samples = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".npz":
            samples.append(_load_npz(path, scenario))
        elif suffix == ".npy":
            samples.append(_load_npy(path, scenario))
        else:
            samples.append(_load_text(path, scenario))
    return samples


def _temporal_pairs(samples: Sequence[CFDSample]) -> list[CFDSample]:
    if len(samples) < 2:
        return list(samples)
    pairs: list[CFDSample] = []
    for previous, current in zip(samples[:-1], samples[1:]):
        if np.shape(previous.targets) != np.shape(current.targets):
            continue
        metadata = dict(current.metadata)
        metadata["input_source_file"] = previous.metadata.get("source_file")
        metadata["temporal_pair"] = True
        pairs.append(
            CFDSample(
                inputs=np.asarray(previous.targets, dtype=np.float32),
                targets=np.asarray(current.targets, dtype=np.float32),
                coordinates=current.coordinates,
                parameters=current.parameters,
                mask=current.mask,
                metadata=metadata,
            )
        )
    return pairs or list(samples)


class CFDBenchDatasetManager:
    """Selective case-archive access for the official CFDBench raw Hub repository."""

    def __init__(
        self,
        token: str | bool | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.hub = HuggingFaceDatasetManager(token=token, endpoint=endpoint)

    @staticmethod
    def scenario_dir(configuration: str) -> str:
        key = configuration.lower().replace("-", "_").replace(" ", "_")
        try:
            return CFDBENCH_SCENARIOS[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown CFDBench configuration {configuration!r}; choose one of "
                f"{sorted(CFDBENCH_SCENARIOS)}"
            ) from exc

    def probe(self, *, revision: str | None = None) -> Any:
        return self.hub.probe(CFDBENCH_REPO, revision=revision)

    def _select_archive(
        self,
        configuration: str,
        *,
        revision: str | None,
        case: str | int | None,
        file_pattern: str | None,
        max_file_size_gb: float | None,
    ) -> tuple[str, int | None]:
        directory = self.scenario_dir(configuration)
        entries = self.hub.list_file_entries(CFDBENCH_REPO, revision=revision)
        candidates = []
        requested = None
        if case is not None:
            case_name = str(case)
            if not case_name.startswith("case"):
                case_name = f"case{case_name}"
            requested = f"{directory}/{case_name}.zip"
        for entry in entries:
            path = str(entry["path"])
            if not path.startswith(f"{directory}/") or not path.lower().endswith(".zip"):
                continue
            if requested and path != requested:
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
                "No CFDBench case archive matched the requested scenario/case/size limit. "
                "Use navier datasets probe cfdbench --configuration <scenario>."
            )
        _, path, size = min(candidates, key=lambda item: (item[0], item[1]))
        return path, size

    def load(
        self,
        configuration: str,
        *,
        split: str = "train",
        revision: str | None = None,
        case: str | int | None = None,
        file_pattern: str | None = None,
        cache_dir: str | Path | None = None,
        max_file_size_gb: float | None = 2.0,
        max_samples: int | None = 64,
        sample_stride: int = 1,
        temporal_pairs: bool = True,
        seed: int = 42,
    ) -> ListCFDDataset:
        selected, _ = self._select_archive(
            configuration,
            revision=revision,
            case=case,
            file_pattern=file_pattern,
            max_file_size_gb=max_file_size_gb,
        )
        cache = resolve_cache_dir(cache_dir, "cfdbench")
        archive = self.hub.download_file(
            CFDBENCH_REPO,
            selected,
            revision=revision,
            cache_dir=cache,
        )
        extraction = cache / "extracted" / Path(selected).stem
        samples = load_cfdbench_archive_samples(
            archive,
            scenario=configuration,
            extract_dir=extraction,
            max_samples=max_samples,
            sample_stride=sample_stride,
        )
        if temporal_pairs:
            samples = _temporal_pairs(samples)
        members = split_members(len(samples), split, seed=seed)
        selected_samples = [samples[index] for index in members]
        resolved_revision = self.hub.resolve_revision(CFDBENCH_REPO, revision)
        plan = ScientificDatasetAccessPlan(
            provider="cfdbench_archive",
            dataset_id="cfdbench",
            configuration=configuration,
            split=split,
            repo_id=CFDBENCH_REPO,
            revision=revision,
            resolved_revision=resolved_revision,
            files=(selected,),
            subset_mode=True,
            official_split=False,
            auth_source=self.hub.auth.source,
            notes=(
                "One selectively downloaded case archive; deterministic within-archive split.",
                "Pickle payloads are never executed.",
            ),
        )
        updated = []
        for sample in selected_samples:
            metadata = dict(sample.metadata)
            metadata["access_plan"] = plan.to_dict()
            metadata["source_split"] = split
            updated.append(
                CFDSample(
                    inputs=sample.inputs,
                    targets=sample.targets,
                    coordinates=sample.coordinates,
                    parameters=sample.parameters,
                    mask=sample.mask,
                    metadata=metadata,
                )
            )
        return ListCFDDataset(updated, access_plan=plan.to_dict())


__all__ = [
    "CFDBENCH_REPO",
    "CFDBENCH_SCENARIOS",
    "CFDBenchDatasetManager",
    "load_cfdbench_archive_samples",
]
