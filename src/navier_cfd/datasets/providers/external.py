from __future__ import annotations

from dataclasses import asdict, dataclass
from fnmatch import fnmatch
import importlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core import CFDSample, DatasetAdapterError
from .common import (
    ListCFDDataset,
    ScientificDatasetAccessPlan,
    coordinate_grid,
    ensure_channel_last_time_series,
    flatten_time_channels,
    split_members,
)


@dataclass(frozen=True)
class LocalDatasetContract:
    dataset_id: str
    provider: str
    source_url: str
    access_mode: str
    representation: str
    temporal: bool
    default_target_fields: tuple[str, ...]
    notes: tuple[str, ...] = ()


LOCAL_DATASET_CONTRACTS: dict[str, LocalDatasetContract] = {
    "airfrans": LocalDatasetContract(
        "airfrans",
        "airfrans_local",
        "https://github.com/Extrality/AirfRANS",
        "official VTK case directories",
        "point_cloud",
        False,
        ("U", "p", "nut"),
        (
            "Download the official AirfRANS Dataset directory before loading.",
            "NAVIER-CFD groups each *_internal.vtu file with its optional *_aerofoil.vtp surface.",
        ),
    ),
    "drivaernetpp": LocalDatasetContract(
        "drivaernetpp",
        "drivaernetpp_local",
        "https://github.com/Mohamedelrefaie/DrivAerNet",
        "Harvard Dataverse/Globus subset staged locally",
        "point_cloud",
        False,
        (
            "pressure",
            "p",
            "cp",
            "coefficient_of_pressure",
            "wall_shear_stress",
            "wss",
            "drag",
            "cd",
        ),
        (
            "The full collection is tens of terabytes; NAVIER-CFD intentionally loads selected local subsets.",
            "Commercial use remains subject to the dataset license.",
        ),
    ),
    "drivaerml": LocalDatasetContract(
        "drivaerml",
        "drivaerml_local",
        "https://caemldatasets.org/drivaerml/",
        "official VTK/OpenFOAM exports staged locally",
        "unstructured",
        True,
        ("U", "velocity", "p", "pressure", "wallShearStress", "wall_shear_stress"),
        ("Use VTK exports or a PyVista-readable OpenFOAM marker for each selected case.",),
    ),
    "scalarflow": LocalDatasetContract(
        "scalarflow",
        "scalarflow_local",
        "https://scalarflow.com/",
        "official reconstruction arrays staged locally",
        "structured",
        True,
        ("density", "velocity", "vel", "smoke"),
        ("NPZ, NPY, HDF5, and exported arrays are supported without loading pickles.",),
    ),
    "shapenet_car": LocalDatasetContract(
        "shapenet_car",
        "shapenet_car_local",
        "https://shapenet.org/",
        "licensed ShapeNet car subset plus optional CFD labels staged locally",
        "point_cloud",
        False,
        ("pressure", "p", "drag", "cd", "target", "targets"),
        (
            "ShapeNet geometry access follows ShapeNet terms.",
            "CFD labels must be supplied by the selected benchmark release or companion metadata.",
        ),
    ),
    "eagle": LocalDatasetContract(
        "eagle",
        "eagle_local",
        "https://arxiv.org/abs/2302.10803",
        "official EAGLE scene tensors/meshes staged locally",
        "hybrid",
        True,
        ("velocity", "pressure", "u", "v", "p", "fields", "target", "targets"),
        ("Tensor files are loaded with safe tensor-only deserialization; arbitrary pickle execution is disabled.",),
    ),
}

SUPPORTED_SUFFIXES = {
    ".npz",
    ".npy",
    ".csv",
    ".txt",
    ".dat",
    ".h5",
    ".hdf5",
    ".vtu",
    ".vtp",
    ".vtk",
    ".ply",
    ".stl",
    ".foam",
    ".pt",
    ".pth",
}


@dataclass(frozen=True)
class LocalDatasetProbe:
    dataset_id: str
    provider: str
    source_url: str
    access_mode: str
    local_path: str | None
    path_exists: bool
    supported_file_count: int
    formats: Mapping[str, int]
    optional_dependencies: Mapping[str, bool]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _optional_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError):
        return False


def _supported_files(root: Path, file_pattern: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    iterator: Iterable[Path] = [root] if root.is_file() else root.rglob("*")
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = path.name if root.is_file() else str(path.relative_to(root))
        if file_pattern and not fnmatch(relative, file_pattern):
            continue
        candidates.append(path)
    return sorted(candidates, key=lambda item: str(item).lower())


def _split_manifest(root: Path, split: str) -> set[str] | None:
    normalized = {"val": "validation", "valid": "validation"}.get(split, split)
    aliases = {
        "train": ("train.txt", "training.txt"),
        "validation": ("validation.txt", "valid.txt", "val.txt"),
        "test": ("test.txt", "testing.txt"),
    }
    if normalized == "all":
        return None
    for directory in (root, root / "splits", root / "train_val_test_splits"):
        for filename in aliases.get(normalized, ()):
            path = directory / filename
            if path.exists():
                values = set()
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    item = line.strip().split(",", 1)[0].strip()
                    if item and not item.startswith("#"):
                        values.add(item)
                        values.add(Path(item).stem)
                return values
    return None


def _select_grouped_items(
    items: Sequence[Any],
    *,
    split: str,
    root: Path,
    key,
    seed: int,
) -> tuple[list[Any], bool]:
    manifest = _split_manifest(root, split)
    if manifest is not None:
        selected = [
            item
            for item in items
            if key(item) in manifest or Path(str(key(item))).stem in manifest
        ]
        if not selected:
            raise DatasetAdapterError(
                f"The {split!r} split manifest exists but matched no local dataset items"
            )
        return selected, True
    indices = split_members(len(items), split, seed=seed)
    return [items[index] for index in indices], False


def _numeric_mapping(mapping: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, value in mapping.items():
        try:
            array = np.asarray(value)
        except Exception:
            continue
        if array.dtype.kind not in "biufc":
            continue
        if array.ndim == 0:
            array = array.reshape(1)
        result[str(key)] = array
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _load_npy(path: Path) -> dict[str, np.ndarray]:
    value = np.load(path, allow_pickle=False)
    if value.dtype.names:
        return {name: np.asarray(value[name]) for name in value.dtype.names}
    if value.ndim == 2 and value.shape[1] >= 3:
        dimensions = min(3, value.shape[1] - 1)
        return {
            "coordinates": value[:, :dimensions],
            "targets": value[:, dimensions:],
        }
    return {"trajectory" if value.ndim >= 3 else "targets": value}


def _load_text(path: Path) -> dict[str, np.ndarray]:
    try:
        structured = np.genfromtxt(
            path,
            names=True,
            dtype=np.float64,
            encoding="utf-8",
            invalid_raise=False,
        )
        if structured.dtype.names:
            return {name: np.asarray(structured[name]) for name in structured.dtype.names}
    except (ValueError, UnicodeDecodeError):
        pass
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] < 2:
        return {"targets": matrix}
    dimensions = min(3, matrix.shape[1] - 1)
    return {
        "coordinates": matrix[:, :dimensions],
        "targets": matrix[:, dimensions:],
    }


def _load_hdf5(path: Path) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "HDF5 local dataset support requires `pip install navier-cfd[scientific-data]`."
        ) from exc

    result: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        def visit(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset) and value.dtype.kind in "biufc":
                result[name] = np.asarray(value)
        handle.visititems(visit)
    return result


def _load_torch(path: Path) -> dict[str, np.ndarray]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Tensor-file support requires `pip install navier-cfd[torch]`."
        ) from exc
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - old torch
        raise RuntimeError(
            "Safe tensor loading requires a PyTorch release with weights_only=True."
        ) from exc
    if hasattr(value, "detach"):
        return {"trajectory": value.detach().cpu().numpy()}
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if hasattr(item, "detach"):
                result[str(key)] = item.detach().cpu().numpy()
            elif isinstance(item, np.ndarray):
                result[str(key)] = item
        if result:
            return result
    raise DatasetAdapterError(
        f"Safe tensor loading found no array-like tensors in {path.name}"
    )


def _load_mesh(path: Path) -> dict[str, np.ndarray]:
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "VTK/OpenFOAM mesh support requires `pip install navier-cfd[mesh-data]`."
        ) from exc
    mesh = pv.read(path)
    if getattr(mesh, "n_points", 0) < 1:
        raise DatasetAdapterError(f"Mesh {path} contains no points")
    if not mesh.point_data and mesh.cell_data:
        mesh = mesh.cell_data_to_point_data()
    result: dict[str, np.ndarray] = {
        "coordinates": np.asarray(mesh.points, dtype=np.float32)
    }
    for name in mesh.point_data:
        array = np.asarray(mesh.point_data[name])
        if array.dtype.kind in "biufc":
            result[str(name)] = array
    return result


def _load_mapping(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        mapping = _load_npz(path)
    elif suffix == ".npy":
        mapping = _load_npy(path)
    elif suffix in {".csv", ".txt", ".dat"}:
        mapping = _load_text(path)
    elif suffix in {".h5", ".hdf5"}:
        mapping = _load_hdf5(path)
    elif suffix in {".vtu", ".vtp", ".vtk", ".ply", ".stl", ".foam"}:
        mapping = _load_mesh(path)
    elif suffix in {".pt", ".pth"}:
        mapping = _load_torch(path)
    else:
        raise DatasetAdapterError(f"Unsupported local dataset file: {path}")
    return _numeric_mapping(mapping)


def _normalized_keys(mapping: Mapping[str, np.ndarray]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in mapping:
        result.setdefault(_normalise(key), key)
        result.setdefault(_normalise(key.rsplit("/", 1)[-1]), key)
    return result


def _lookup(mapping: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray | None:
    normalized = _normalized_keys(mapping)
    for name in names:
        key = normalized.get(_normalise(name))
        if key is not None:
            return np.asarray(mapping[key])
    return None


def _matching_fields(
    mapping: Mapping[str, np.ndarray],
    requested: Sequence[str],
) -> list[tuple[str, np.ndarray]]:
    normalized = _normalized_keys(mapping)
    fields = []
    for name in requested:
        key = normalized.get(_normalise(name))
        if key is not None and all(existing != key for existing, _ in fields):
            fields.append((key, np.asarray(mapping[key])))
    return fields


def _coordinates(mapping: Mapping[str, np.ndarray]) -> np.ndarray | None:
    direct = _lookup(mapping, ("coordinates", "coords", "points", "pos", "vertices", "xyz"))
    if direct is not None:
        return direct.astype(np.float32, copy=False)
    components = []
    for name in ("x", "y", "z", "x_coordinate", "y_coordinate", "z_coordinate"):
        value = _lookup(mapping, (name,))
        if value is not None and value.ndim == 1:
            components.append(value)
    if len(components) >= 2 and all(component.shape == components[0].shape for component in components):
        return np.stack(components[:3], axis=-1).astype(np.float32)
    return None


def _combine_arrays(fields: Sequence[tuple[str, np.ndarray]]) -> tuple[np.ndarray, tuple[str, ...]]:
    if not fields:
        raise DatasetAdapterError("No compatible target fields were found")
    arrays = []
    names = []
    reference_shape: tuple[int, ...] | None = None
    for name, raw in fields:
        array = np.asarray(raw)
        if array.ndim == 1:
            array = array[..., None]
        if reference_shape is None:
            reference_shape = array.shape[:-1]
        if array.shape[:-1] != reference_shape:
            continue
        arrays.append(array)
        names.append(name)
    if not arrays:
        raise DatasetAdapterError("Selected target fields do not share a compatible sample shape")
    return np.concatenate(arrays, axis=-1), tuple(names)


def _steady_sample(
    mapping: Mapping[str, np.ndarray],
    *,
    contract: LocalDatasetContract,
    path: Path,
    target_fields: Sequence[str] | None,
    input_fields: Sequence[str] | None,
    access_plan: Mapping[str, Any],
) -> CFDSample:
    coords = _coordinates(mapping)
    selected_targets = _matching_fields(
        mapping,
        target_fields or contract.default_target_fields,
    )
    if not selected_targets:
        explicit = _lookup(mapping, ("targets", "target", "y", "fields", "solution"))
        if explicit is not None:
            selected_targets = [("targets", explicit)]
    if not selected_targets:
        excluded = {
            _normalise(name)
            for name in (
                "coordinates",
                "coords",
                "points",
                "pos",
                "vertices",
                "x",
                "y",
                "z",
                "x_coordinate",
                "y_coordinate",
                "z_coordinate",
                "inputs",
                "input",
            )
        }
        selected_targets = [
            (name, value)
            for name, value in mapping.items()
            if _normalise(name.rsplit("/", 1)[-1]) not in excluded and value.ndim >= 1
        ]
    targets, target_names = _combine_arrays(selected_targets)

    if input_fields:
        selected_inputs = _matching_fields(mapping, input_fields)
        inputs, input_names = _combine_arrays(selected_inputs)
    else:
        explicit_inputs = _lookup(mapping, ("inputs", "input", "features"))
        if explicit_inputs is not None and (
            coords is None or np.shape(explicit_inputs) != np.shape(coords)
        ):
            inputs = np.asarray(explicit_inputs)
            if inputs.ndim == 1:
                inputs = inputs[..., None]
            input_names = ("inputs",)
        elif coords is not None:
            inputs = coords.copy()
            input_names = ("coordinates",)
        else:
            inputs = np.arange(targets.shape[0], dtype=np.float32)[:, None]
            input_names = ("index",)
            coords = inputs.copy()

    return CFDSample(
        inputs=np.asarray(inputs, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        coordinates=None if coords is None else np.asarray(coords, dtype=np.float32),
        metadata={
            "dataset_id": contract.dataset_id,
            "provider": contract.provider,
            "source_file": path.name,
            "representation": contract.representation,
            "input_fields": input_names,
            "target_fields": target_names,
            "access_plan": dict(access_plan),
        },
    )


def _temporal_fields(
    mapping: Mapping[str, np.ndarray],
    contract: LocalDatasetContract,
    target_fields: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]] | None:
    requested = target_fields or contract.default_target_fields
    fields = _matching_fields(mapping, requested)
    if not fields:
        trajectory = _lookup(
            mapping,
            ("trajectory", "trajectories", "sequence", "states", "fields", "targets"),
        )
        if trajectory is not None:
            return np.asarray(trajectory), ("trajectory",)
        return None

    candidates = []
    for name, value in fields:
        array = np.asarray(value)
        if array.ndim >= 3:
            if array.ndim >= 3 and array.shape[-1] > 16 and array.shape[1] <= 16:
                array = np.moveaxis(array, 1, -1)
            elif array.shape[-1] > 16:
                array = array[..., None]
            candidates.append((name, array))
    if not candidates:
        return None
    reference = candidates[0][1].shape[:-1]
    compatible = [(name, value) for name, value in candidates if value.shape[:-1] == reference]
    trajectory = np.concatenate([value for _, value in compatible], axis=-1)
    return trajectory, tuple(name for name, _ in compatible)


def _temporal_samples(
    mapping: Mapping[str, np.ndarray],
    *,
    contract: LocalDatasetContract,
    path: Path,
    target_fields: Sequence[str] | None,
    n_steps_input: int,
    n_steps_output: int,
    time_stride: int,
    window_stride: int,
    max_windows: int | None,
    access_plan: Mapping[str, Any],
) -> list[CFDSample]:
    if max_windows is not None and max_windows <= 0:
        return []
    selected = _temporal_fields(mapping, contract, target_fields)
    if selected is None:
        return []
    raw, field_names = selected
    if raw.ndim < 3:
        return []
    if raw.shape[-1] > 16:
        trajectory = ensure_channel_last_time_series(raw, time_axis=0)
    else:
        trajectory = np.asarray(raw)
    horizon = (n_steps_input + n_steps_output - 1) * time_stride + 1
    coords = _coordinates(mapping)
    if coords is None and trajectory.ndim >= 3:
        spatial_shape = trajectory.shape[1:-1]
        coords = coordinate_grid(
            [np.linspace(0.0, 1.0, size, dtype=np.float32) for size in spatial_shape]
        )
    samples = []
    for start in range(
        0,
        max(0, trajectory.shape[0] - horizon + 1),
        max(1, window_stride),
    ):
        indices = start + np.arange(n_steps_input + n_steps_output) * time_stride
        selected_window = trajectory[indices]
        samples.append(
            CFDSample(
                inputs=flatten_time_channels(selected_window[:n_steps_input]).astype(
                    np.float32, copy=False
                ),
                targets=flatten_time_channels(selected_window[n_steps_input:]).astype(
                    np.float32, copy=False
                ),
                coordinates=coords,
                metadata={
                    "dataset_id": contract.dataset_id,
                    "provider": contract.provider,
                    "source_file": path.name,
                    "window_start": start,
                    "input_steps": n_steps_input,
                    "output_steps": n_steps_output,
                    "time_stride": time_stride,
                    "target_fields": field_names,
                    "representation": contract.representation,
                    "access_plan": dict(access_plan),
                },
            )
        )
        if max_windows is not None and len(samples) >= max_windows:
            break
    return samples


def _airfrans_case_sample(
    case_dir: Path,
    internal_path: Path,
    *,
    contract: LocalDatasetContract,
    access_plan: Mapping[str, Any],
) -> CFDSample:
    internal = _load_mesh(internal_path)
    case_name = internal_path.name.removesuffix("_internal.vtu")
    surface_path = case_dir / f"{case_name}_aerofoil.vtp"
    surface = _load_mesh(surface_path) if surface_path.exists() else None

    internal_coords = _coordinates(internal)
    if internal_coords is None:
        raise DatasetAdapterError(f"AirfRANS internal mesh {internal_path} has no coordinates")
    n_internal = len(internal_coords)
    surface_coords = _coordinates(surface or {})
    n_surface = 0 if surface_coords is None else len(surface_coords)

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", case_name)
    speed = float(numbers[-2]) if len(numbers) >= 2 else 1.0
    alpha = np.deg2rad(float(numbers[-1])) if numbers else 0.0
    free_stream = np.array([np.cos(alpha) * speed, np.sin(alpha) * speed], dtype=np.float32)

    def vector(mapping: Mapping[str, np.ndarray], names: Sequence[str], count: int, width: int) -> np.ndarray:
        value = _lookup(mapping, names)
        if value is None:
            return np.zeros((count, width), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.shape[-1] < width:
            array = np.pad(array, ((0, 0), (0, width - array.shape[-1])))
        return array[:, :width]

    internal_u = vector(internal, ("U", "velocity"), n_internal, 2)
    internal_p = vector(internal, ("p", "pressure"), n_internal, 1)
    internal_nut = vector(internal, ("nut", "turbulent_viscosity"), n_internal, 1)
    internal_sdf = vector(internal, ("implicit_distance", "sdf"), n_internal, 1)
    internal_normal = np.zeros((n_internal, 2), dtype=np.float32)
    internal_inputs = np.concatenate(
        [np.broadcast_to(free_stream, (n_internal, 2)), -internal_sdf, internal_normal],
        axis=-1,
    )
    internal_targets = np.concatenate([internal_u, internal_p, internal_nut], axis=-1)

    inputs = internal_inputs
    targets = internal_targets
    coordinates = internal_coords[:, :2]
    surface_mask = np.zeros(n_internal, dtype=bool)
    if surface is not None and surface_coords is not None:
        surface_u = vector(surface, ("U", "velocity"), n_surface, 2)
        surface_p = vector(surface, ("p", "pressure"), n_surface, 1)
        surface_nut = vector(surface, ("nut", "turbulent_viscosity"), n_surface, 1)
        surface_normal = vector(surface, ("Normals", "normals", "normal"), n_surface, 2)
        surface_inputs = np.concatenate(
            [
                np.broadcast_to(free_stream, (n_surface, 2)),
                np.zeros((n_surface, 1), dtype=np.float32),
                -surface_normal,
            ],
            axis=-1,
        )
        surface_targets = np.concatenate([surface_u, surface_p, surface_nut], axis=-1)
        inputs = np.concatenate([inputs, surface_inputs], axis=0)
        targets = np.concatenate([targets, surface_targets], axis=0)
        coordinates = np.concatenate([coordinates, surface_coords[:, :2]], axis=0)
        surface_mask = np.concatenate(
            [surface_mask, np.ones(n_surface, dtype=bool)],
            axis=0,
        )

    return CFDSample(
        inputs=inputs,
        targets=targets,
        coordinates=coordinates,
        parameters={"free_stream_speed": speed, "angle_of_attack_deg": float(np.rad2deg(alpha))},
        mask=surface_mask,
        metadata={
            "dataset_id": "airfrans",
            "provider": contract.provider,
            "case": case_name,
            "source_file": internal_path.name,
            "surface_file": surface_path.name if surface_path.exists() else None,
            "representation": contract.representation,
            "input_fields": ("u_infinity_x", "u_infinity_y", "signed_distance", "normal_x", "normal_y"),
            "target_fields": ("u", "v", "p", "nut"),
            "access_plan": dict(access_plan),
        },
    )


class LocalScientificDatasetManager:
    """Load catalogued scientific datasets from official local exports."""

    def contract(self, dataset_id: str) -> LocalDatasetContract:
        try:
            return LOCAL_DATASET_CONTRACTS[dataset_id]
        except KeyError as exc:
            raise KeyError(
                f"No local scientific provider contract for {dataset_id!r}; "
                f"available: {sorted(LOCAL_DATASET_CONTRACTS)}"
            ) from exc

    def probe(
        self,
        dataset_id: str,
        *,
        local_path: str | Path | None = None,
        file_pattern: str | None = None,
    ) -> LocalDatasetProbe:
        contract = self.contract(dataset_id)
        path = None if local_path is None else Path(local_path).expanduser()
        exists = bool(path and path.exists())
        files = _supported_files(path, file_pattern) if exists and path is not None else []
        formats: dict[str, int] = {}
        for file in files:
            formats[file.suffix.lower()] = formats.get(file.suffix.lower(), 0) + 1
        return LocalDatasetProbe(
            dataset_id=dataset_id,
            provider=contract.provider,
            source_url=contract.source_url,
            access_mode=contract.access_mode,
            local_path=None if path is None else str(path),
            path_exists=exists,
            supported_file_count=len(files),
            formats=dict(sorted(formats.items())),
            optional_dependencies={
                "h5py": _optional_available("h5py"),
                "pyvista": _optional_available("pyvista"),
                "torch": _optional_available("torch"),
            },
            notes=contract.notes,
        )

    def load(
        self,
        dataset_id: str,
        *,
        local_path: str | Path,
        configuration: str | None = None,
        split: str = "train",
        file_pattern: str | None = None,
        max_samples: int | None = 64,
        sample_stride: int = 1,
        target_fields: Sequence[str] | None = None,
        input_fields: Sequence[str] | None = None,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        time_stride: int = 1,
        window_stride: int = 1,
        max_windows: int | None = 128,
        seed: int = 42,
    ) -> ListCFDDataset:
        contract = self.contract(dataset_id)
        root = Path(local_path).expanduser()
        if not root.exists():
            raise FileNotFoundError(
                f"Local path {root} does not exist. Obtain the dataset from {contract.source_url} "
                "and pass local_path to NAVIER-CFD."
            )

        if dataset_id == "airfrans":
            internal_files = sorted(root.rglob("*_internal.vtu"))
            if not internal_files:
                raise DatasetAdapterError(
                    "No AirfRANS *_internal.vtu files were found under the local path"
                )
            selected_files, official_split = _select_grouped_items(
                internal_files,
                split=split,
                root=root,
                key=lambda item: item.name.removesuffix("_internal.vtu"),
                seed=seed,
            )
            selected_files = selected_files[:: max(1, sample_stride)]
            if max_samples is not None:
                selected_files = selected_files[:max_samples]
            plan = ScientificDatasetAccessPlan(
                provider=contract.provider,
                dataset_id=dataset_id,
                configuration=configuration or "airfoil",
                split=split,
                repo_id=contract.source_url,
                revision=None,
                resolved_revision=None,
                files=tuple(str(path.relative_to(root)) for path in selected_files),
                subset_mode=max_samples is not None,
                official_split=official_split,
                auth_source="local",
                notes=contract.notes,
            )
            samples = [
                _airfrans_case_sample(
                    path.parent,
                    path,
                    contract=contract,
                    access_plan=plan.to_dict(),
                )
                for path in selected_files
            ]
            return ListCFDDataset(samples, access_plan=plan.to_dict())

        files = _supported_files(root, file_pattern)
        if not files:
            raise DatasetAdapterError(
                f"No supported local files were found for {dataset_id}. "
                f"Supported suffixes: {sorted(SUPPORTED_SUFFIXES)}"
            )
        selected_files, official_split = _select_grouped_items(
            files,
            split=split,
            root=root if root.is_dir() else root.parent,
            key=lambda item: str(item.relative_to(root)) if root.is_dir() else item.name,
            seed=seed,
        )
        selected_files = selected_files[:: max(1, sample_stride)]
        if max_samples is not None:
            selected_files = selected_files[:max_samples]
        plan = ScientificDatasetAccessPlan(
            provider=contract.provider,
            dataset_id=dataset_id,
            configuration=configuration or "default",
            split=split,
            repo_id=contract.source_url,
            revision=None,
            resolved_revision=None,
            files=tuple(
                str(path.relative_to(root)) if root.is_dir() else path.name
                for path in selected_files
            ),
            subset_mode=max_samples is not None or max_windows is not None,
            official_split=official_split,
            auth_source="local",
            notes=contract.notes,
        )

        samples: list[CFDSample] = []
        for path in selected_files:
            mapping = _load_mapping(path)
            temporal = (
                _temporal_samples(
                    mapping,
                    contract=contract,
                    path=path,
                    target_fields=target_fields,
                    n_steps_input=n_steps_input,
                    n_steps_output=n_steps_output,
                    time_stride=time_stride,
                    window_stride=window_stride,
                    max_windows=None if max_windows is None else max(0, max_windows - len(samples)),
                    access_plan=plan.to_dict(),
                )
                if contract.temporal
                else []
            )
            if temporal:
                samples.extend(temporal)
            else:
                samples.append(
                    _steady_sample(
                        mapping,
                        contract=contract,
                        path=path,
                        target_fields=target_fields,
                        input_fields=input_fields,
                        access_plan=plan.to_dict(),
                    )
                )
            if max_windows is not None and len(samples) >= max_windows:
                samples = samples[:max_windows]
                break
        if not samples:
            raise DatasetAdapterError(f"No canonical samples could be produced for {dataset_id}")
        return ListCFDDataset(samples, access_plan=plan.to_dict())


__all__ = [
    "LOCAL_DATASET_CONTRACTS",
    "LocalDatasetContract",
    "LocalDatasetProbe",
    "LocalScientificDatasetManager",
    "SUPPORTED_SUFFIXES",
]
