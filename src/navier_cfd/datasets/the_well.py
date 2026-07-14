from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core import CFDSample, DatasetAdapterError


THE_WELL_HF_BASE = "hf://datasets/polymathic-ai/"

KNOWN_WELL_DATASETS = (
    "acoustic_scattering_discontinuous",
    "acoustic_scattering_inclusions",
    "acoustic_scattering_maze",
    "active_matter",
    "convective_envelope_rsg",
    "euler_multi_quadrants_openBC",
    "euler_multi_quadrants_periodicBC",
    "gray_scott_reaction_diffusion",
    "helmholtz_staircase",
    "MHD_64",
    "MHD_256",
    "planetswe",
    "post_neutron_star_merger",
    "rayleigh_benard",
    "rayleigh_benard_uniform",
    "rayleigh_taylor_instability",
    "shear_flow",
    "supernova_explosion_64",
    "supernova_explosion_128",
    "turbulence_gravity_cooling",
    "turbulent_radiative_layer_2D",
    "turbulent_radiative_layer_3D",
    "viscoelastic_instability_v2",
)


class MissingTheWellDependency(RuntimeError):
    """Raised when official The Well support is requested without its package."""


@dataclass(frozen=True)
class TheWellAccessPlan:
    dataset_name: str
    split: str
    base_path: str
    n_steps_input: int
    n_steps_output: int
    use_normalization: bool
    normalization_type: str | None
    full_trajectory_mode: bool
    provider_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": "the_well",
            "access_backend": "the_well.data.WellDataset",
            "dataset_name": self.dataset_name,
            "split": self.split,
            "base_path": self.base_path,
            "n_steps_input": self.n_steps_input,
            "n_steps_output": self.n_steps_output,
            "use_normalization": self.use_normalization,
            "normalization_type": self.normalization_type,
            "full_trajectory_mode": self.full_trajectory_mode,
            "provider_version": self.provider_version,
        }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten_time_channels(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim < 3:
        raise DatasetAdapterError("The Well fields must have shape [time, spatial..., channels]")
    moved = np.moveaxis(array, 0, -2)
    return moved.reshape(*moved.shape[:-2], moved.shape[-2] * moved.shape[-1])


def _space_grid(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = list(value.values())
    if isinstance(value, (tuple, list)):
        vectors = [_to_numpy(item) for item in value]
        if vectors and all(vector.ndim == 1 for vector in vectors):
            return np.stack(np.meshgrid(*vectors, indexing="ij"), axis=-1).astype(np.float32)
    array = _to_numpy(value)
    if array.ndim >= 2:
        return array.astype(np.float32, copy=False)
    return None


def _metadata_value(metadata: Any, name: str, default: Any = None) -> Any:
    if metadata is None:
        return default
    if isinstance(metadata, Mapping):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


def _ordered_names(metadata: Any, name: str, channels: int) -> tuple[str, ...]:
    grouped = _metadata_value(metadata, name, {})
    names: list[str] = []
    if isinstance(grouped, Mapping):
        for order in sorted(grouped, key=lambda value: str(value)):
            names.extend(str(item) for item in grouped[order])
    elif isinstance(grouped, Sequence) and not isinstance(grouped, (str, bytes)):
        names.extend(str(item) for item in grouped)
    if len(names) < channels:
        prefix = "field" if name == "field_names" else "constant_field"
        names.extend(f"{prefix}_{index}" for index in range(len(names), channels))
    return tuple(names[:channels])


def _named_vector(value: Any, names: Sequence[str], prefix: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    array = _to_numpy(value)
    if array.size == 0:
        return {}
    flat = array.reshape(-1)
    resolved = list(names)
    if len(resolved) < len(flat):
        resolved.extend(f"{prefix}_{index}" for index in range(len(resolved), len(flat)))
    return {resolved[index]: flat[index] for index in range(len(flat))}


def _type_name(value: Any | None) -> str | None:
    if value is None:
        return None
    return getattr(value, "__name__", value.__class__.__name__)


class TheWellDatasetAdapter:
    """Map official ``WellDataset`` records to channel-last :class:`CFDSample` objects.

    Official records store variable fields as ``[time, spatial..., channels]``. By
    default, NAVIER-CFD flattens history and field channels into the final axis and
    concatenates constant fields, while preserving field semantics in metadata.
    """

    def __init__(
        self,
        raw_dataset: Any,
        *,
        dataset_name: str,
        split: str,
        flatten_time_to_channels: bool = True,
        include_constant_fields: bool = True,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.dataset_name = dataset_name
        self.split = split
        self.flatten_time_to_channels = flatten_time_to_channels
        self.include_constant_fields = include_constant_fields

    def adapt(self, record: Mapping[str, Any], *, index: int | None = None) -> CFDSample:
        if "input_fields" not in record or "output_fields" not in record:
            raise DatasetAdapterError("Official The Well records must contain input_fields and output_fields")
        raw_inputs = _to_numpy(record["input_fields"])
        raw_targets = _to_numpy(record["output_fields"])
        provider_metadata = getattr(self.raw_dataset, "metadata", None)
        field_names = _ordered_names(provider_metadata, "field_names", raw_inputs.shape[-1])

        constant_fields = record.get("constant_fields")
        constant_array = None
        if constant_fields is not None:
            constant_array = _to_numpy(constant_fields)
            if constant_array.size == 0:
                constant_array = None

        if self.flatten_time_to_channels:
            inputs = _flatten_time_channels(raw_inputs)
            targets = _flatten_time_channels(raw_targets)
            layout = "spatial_channel_last_time_flattened"
            input_channel_names = tuple(
                f"t{time_index}:{field_name}"
                for time_index in range(raw_inputs.shape[0])
                for field_name in field_names
            )
            if self.include_constant_fields and constant_array is not None:
                if constant_array.shape[:-1] != inputs.shape[:-1]:
                    raise DatasetAdapterError(
                        "The Well constant_fields spatial shape does not match input_fields"
                    )
                inputs = np.concatenate((inputs, constant_array), axis=-1)
                constant_names = _ordered_names(
                    provider_metadata,
                    "constant_field_names",
                    constant_array.shape[-1],
                )
                input_channel_names = input_channel_names + constant_names
        else:
            inputs = raw_inputs
            targets = raw_targets
            layout = "time_spatial_channel_last"
            input_channel_names = field_names
            if self.include_constant_fields and constant_array is not None:
                raise DatasetAdapterError(
                    "include_constant_fields=True requires flatten_time_to_channels=True"
                )

        coordinates = _space_grid(record.get("space_grid"))
        raw_mask = record.get("valid_mask")
        mask = _to_numpy(raw_mask).astype(bool) if raw_mask is not None else None
        constant_scalar_names = tuple(_metadata_value(provider_metadata, "constant_scalar_names", ()))
        scalar_names = tuple(_metadata_value(provider_metadata, "scalar_names", ()))
        parameters = _named_vector(
            record.get("constant_scalars"),
            constant_scalar_names,
            "constant_scalar",
        )

        metadata: dict[str, Any] = {
            "dataset_id": "the_well",
            "provider": "the_well",
            "well_dataset_name": self.dataset_name,
            "source_split": self.split,
            "representation": "structured",
            "layout": layout,
            "field_names": field_names,
            "input_channel_names": input_channel_names,
            "input_steps": int(raw_inputs.shape[0]),
            "output_steps": int(raw_targets.shape[0]),
            "physical_field_channels": int(raw_inputs.shape[-1]),
            "input_time_grid": record.get("input_time_grid"),
            "output_time_grid": record.get("output_time_grid"),
            "input_scalars": record.get("input_scalars"),
            "output_scalars": record.get("output_scalars"),
            "scalar_names": scalar_names,
            "boundary_conditions": record.get("boundary_conditions"),
            "normalization": {
                "enabled": bool(getattr(self.raw_dataset, "use_normalization", False)),
                "type": _type_name(getattr(self.raw_dataset, "normalization_type", None)),
                "source": "official_the_well_statistics",
            },
        }
        for name in ("spatial_resolution", "n_spatial_dims", "n_fields", "dataset_name", "grid_type"):
            value = _metadata_value(provider_metadata, name)
            if value is not None:
                metadata[name] = value
        if index is not None:
            metadata["index"] = index

        return CFDSample(
            inputs=inputs.astype(np.float32, copy=False),
            targets=targets.astype(np.float32, copy=False),
            coordinates=coordinates,
            parameters=parameters,
            mask=mask,
            metadata=metadata,
        )


class TheWellAdaptedDataset:
    def __init__(self, raw_dataset: Any, adapter: TheWellDatasetAdapter) -> None:
        self.raw_dataset = raw_dataset
        self.adapter = adapter
        self.metadata = getattr(raw_dataset, "metadata", None)
        self.access_plan = getattr(raw_dataset, "navier_access_plan", None)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> CFDSample:
        record = self.raw_dataset[index]
        if not isinstance(record, Mapping):
            raise DatasetAdapterError("The Well dataset items must be mapping-like records")
        return self.adapter.adapt(record, index=index)


class TheWellDatasetManager:
    """Official-provider access for The Well datasets.

    The Well is not routed through ``datasets.load_dataset('polymathic-ai/the_well')``.
    Streaming uses fsspec through the official ``WellDataset`` with
    ``hf://datasets/polymathic-ai/`` as the provider base path.
    """

    hf_base_path = THE_WELL_HF_BASE

    @staticmethod
    def _dataset_class() -> Any:
        try:
            from the_well.data import WellDataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingTheWellDependency(
                "Official The Well support requires `pip install navier-cfd[the-well]`."
            ) from exc
        return WellDataset

    @staticmethod
    def _default_normalization_type() -> Any:
        try:
            from the_well.data.normalization import ZScoreNormalization
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingTheWellDependency(
                "The Well normalization requires `pip install navier-cfd[the-well]`."
            ) from exc
        return ZScoreNormalization

    @staticmethod
    def _download_function() -> Any:
        try:
            from the_well.utils.download import well_download
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingTheWellDependency(
                "Downloading The Well requires `pip install navier-cfd[the-well]`."
            ) from exc
        return well_download

    @staticmethod
    def provider_version() -> str | None:
        try:
            return importlib_metadata.version("the_well")
        except importlib_metadata.PackageNotFoundError:
            return None

    def list_datasets(self) -> tuple[str, ...]:
        try:
            from the_well.data.utils import WELL_DATASETS
        except ImportError:
            return KNOWN_WELL_DATASETS
        return tuple(str(name) for name in WELL_DATASETS)

    def load(
        self,
        dataset_name: str,
        *,
        split: str = "train",
        base_path: str | Path | None = None,
        streaming: bool = True,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        use_normalization: bool = False,
        normalization_type: Any | None = None,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        flatten_tensors: bool = True,
        return_grid: bool = True,
        boundary_return_type: str | None = "padding",
        full_trajectory_mode: bool = False,
        storage_options: Mapping[str, Any] | None = None,
        adapt: bool = True,
        flatten_time_to_channels: bool = True,
        include_constant_fields: bool = True,
        **kwargs: Any,
    ) -> Any:
        if split not in {"train", "valid", "test"}:
            raise ValueError("The Well split must be 'train', 'valid', or 'test'")
        if not dataset_name.strip():
            raise ValueError("dataset_name is required for The Well")
        resolved_base = str(base_path or self.hf_base_path)
        if not streaming and base_path is None:
            raise ValueError("A local base_path is required when streaming=False")
        if use_normalization and normalization_type is None:
            normalization_type = self._default_normalization_type()
        if not use_normalization:
            normalization_type = None

        constructor: dict[str, Any] = {
            "well_base_path": resolved_base,
            "well_dataset_name": dataset_name,
            "well_split_name": split,
            "n_steps_input": n_steps_input,
            "n_steps_output": n_steps_output,
            "use_normalization": use_normalization,
            "normalization_type": normalization_type,
            "min_dt_stride": min_dt_stride,
            "max_dt_stride": max_dt_stride,
            "flatten_tensors": flatten_tensors,
            "return_grid": return_grid,
            "boundary_return_type": boundary_return_type,
            "full_trajectory_mode": full_trajectory_mode,
            "storage_options": dict(storage_options or {}) or None,
            **kwargs,
        }
        raw = self._dataset_class()(**constructor)
        plan = TheWellAccessPlan(
            dataset_name=dataset_name,
            split=split,
            base_path=resolved_base,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=use_normalization,
            normalization_type=_type_name(normalization_type),
            full_trajectory_mode=full_trajectory_mode,
            provider_version=self.provider_version(),
        )
        raw.navier_access_plan = plan.to_dict()
        if not adapt:
            return raw
        adapter = TheWellDatasetAdapter(
            raw,
            dataset_name=dataset_name,
            split=split,
            flatten_time_to_channels=flatten_time_to_channels,
            include_constant_fields=include_constant_fields,
        )
        return TheWellAdaptedDataset(raw, adapter)

    def download(
        self,
        dataset_name: str,
        *,
        base_path: str | Path,
        split: str | None = None,
        first_only: bool = False,
        parallel: bool = False,
    ) -> Any:
        if split is not None and split not in {"train", "valid", "test"}:
            raise ValueError("The Well split must be 'train', 'valid', or 'test'")
        return self._download_function()(
            base_path=str(base_path),
            dataset=dataset_name,
            split=split,
            first_only=first_only,
            parallel=parallel,
        )


def load_the_well(dataset_name: str, **kwargs: Any) -> Any:
    return TheWellDatasetManager().load(dataset_name, **kwargs)


__all__ = [
    "KNOWN_WELL_DATASETS",
    "MissingTheWellDependency",
    "THE_WELL_HF_BASE",
    "TheWellAccessPlan",
    "TheWellAdaptedDataset",
    "TheWellDatasetAdapter",
    "TheWellDatasetManager",
    "load_the_well",
]
