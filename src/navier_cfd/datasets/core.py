from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np


class DatasetAdapterError(ValueError):
    """Raised when a raw dataset record cannot be converted to the canonical schema."""


@dataclass(frozen=True)
class CFDSample:
    """Canonical sample shared by structured, mesh, point-cloud, and particle data.

    Arrays use channel-last convention. Structured fields may have shape
    ``[*spatial, channels]``; unstructured fields use ``[points, channels]``.
    """

    inputs: Any
    targets: Any
    coordinates: Any | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    mask: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CFDBatch:
    inputs: Any
    targets: Any
    coordinates: Any | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    mask: Any | None = None
    metadata: tuple[Mapping[str, Any], ...] = ()

    def to(self, device: Any) -> "CFDBatch":
        def move(value: Any) -> Any:
            return value.to(device) if hasattr(value, "to") else value

        return CFDBatch(
            inputs=move(self.inputs),
            targets=move(self.targets),
            coordinates=move(self.coordinates) if self.coordinates is not None else None,
            parameters={key: move(value) for key, value in self.parameters.items()},
            mask=move(self.mask) if self.mask is not None else None,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class DatasetProfile:
    dataset_id: str
    representation: str
    input_aliases: tuple[str, ...]
    target_aliases: tuple[str, ...]
    coordinate_aliases: tuple[str, ...] = ()
    mask_aliases: tuple[str, ...] = ()
    parameter_aliases: tuple[str, ...] = ()
    notes: str = ""


BUILTIN_DATASET_PROFILES: dict[str, DatasetProfile] = {
    "pdebench": DatasetProfile(
        "pdebench",
        "structured",
        ("input", "inputs", "x", "initial_condition", "u0", "state_in", "features"),
        ("target", "targets", "y", "solution", "state", "state_out", "fields"),
        ("coordinates", "coords", "grid", "points", "xyz"),
        parameter_aliases=("parameters", "params", "pde_parameters"),
    ),
    "cfdbench": DatasetProfile(
        "cfdbench",
        "structured",
        ("input", "inputs", "x", "bc", "boundary", "initial", "features"),
        ("target", "targets", "y", "solution", "velocity", "fields"),
        ("coordinates", "coords", "grid", "mesh"),
        ("mask", "fluid_mask", "domain_mask"),
        ("parameters", "params", "reynolds", "Re"),
    ),
    "realpdebench": DatasetProfile(
        "realpdebench",
        "structured",
        ("input", "inputs", "x", "history", "simulation", "features"),
        ("target", "targets", "y", "future", "measurement", "real", "solution"),
        ("coordinates", "coords", "grid", "points"),
        ("mask", "valid_mask"),
        ("parameters", "params", "control"),
    ),
    "the_well": DatasetProfile(
        "the_well",
        "structured",
        ("input", "inputs", "x", "input_fields", "history", "state_in"),
        ("target", "targets", "y", "output_fields", "future", "state_out"),
        ("coordinates", "coords", "grid"),
        ("mask", "valid_mask"),
        ("parameters", "params", "scalars"),
    ),
    "apebench": DatasetProfile(
        "apebench",
        "structured",
        ("input", "inputs", "x", "u0", "state_in"),
        ("target", "targets", "y", "trajectory", "state_out"),
        ("coordinates", "coords", "grid"),
        parameter_aliases=("parameters", "params", "pde_config"),
    ),
    "scalarflow": DatasetProfile(
        "scalarflow",
        "structured",
        ("input", "inputs", "x", "density_history", "density_in"),
        ("target", "targets", "y", "density", "density_out", "future"),
        ("coordinates", "coords", "grid"),
        ("mask", "valid_mask"),
        ("parameters", "params"),
    ),
    "airfrans": DatasetProfile(
        "airfrans",
        "point_cloud",
        ("input", "inputs", "x", "features", "node_features", "geometry"),
        ("target", "targets", "y", "surface", "volume", "fields"),
        ("coordinates", "coords", "pos", "points", "mesh_points"),
        ("mask", "surface_mask", "volume_mask"),
        ("parameters", "params", "inlet", "conditions"),
    ),
    "drivaernetpp": DatasetProfile(
        "drivaernetpp",
        "point_cloud",
        ("input", "inputs", "x", "features", "geometry", "surface_points"),
        ("target", "targets", "y", "pressure", "wall_shear", "fields"),
        ("coordinates", "coords", "pos", "points", "surface_points"),
        ("mask", "surface_mask"),
        ("parameters", "params", "conditions"),
    ),
    "drivaerml": DatasetProfile(
        "drivaerml",
        "unstructured",
        ("input", "inputs", "x", "features", "geometry", "mesh_features"),
        ("target", "targets", "y", "pressure", "velocity", "fields"),
        ("coordinates", "coords", "pos", "points", "vertices"),
        ("mask", "surface_mask", "volume_mask"),
        ("parameters", "params", "conditions"),
    ),
    "shapenet_car": DatasetProfile(
        "shapenet_car",
        "point_cloud",
        ("input", "inputs", "x", "features", "geometry", "points"),
        ("target", "targets", "y", "pressure", "drag", "fields"),
        ("coordinates", "coords", "pos", "points"),
        ("mask", "surface_mask"),
        ("parameters", "params"),
    ),
    "eagle": DatasetProfile(
        "eagle",
        "hybrid",
        ("input", "inputs", "x", "features", "history", "node_features"),
        ("target", "targets", "y", "future", "fields", "solution"),
        ("coordinates", "coords", "pos", "grid", "points"),
        ("mask", "valid_mask", "domain_mask"),
        ("parameters", "params", "conditions"),
    ),
}


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _lookup(record: Mapping[str, Any], aliases: Sequence[str]) -> Any | None:
    lower = {str(key).lower(): key for key in record}
    for alias in aliases:
        if alias in record:
            return record[alias]
        actual = lower.get(alias.lower())
        if actual is not None:
            return record[actual]
    return None


def _combine_fields(value: Any, names: Sequence[str] | None = None) -> Any:
    if not isinstance(value, Mapping):
        return value
    keys = list(names or value.keys())
    arrays = []
    for key in keys:
        if key not in value:
            continue
        array = _to_numpy(value[key])
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim == 1:
            array = array[..., None]
        arrays.append(array)
    if not arrays:
        raise DatasetAdapterError("No requested fields were found in the mapping container")
    try:
        return np.concatenate(arrays, axis=-1)
    except ValueError as exc:
        shapes = [array.shape for array in arrays]
        raise DatasetAdapterError(f"Cannot concatenate field arrays with shapes {shapes}") from exc


class DatasetAdapter:
    """Converts raw records into :class:`CFDSample` objects.

    Alias-based defaults cover the 11 registered benchmark families. Exact field
    names may be overridden for a specific release without changing model code.
    """

    def __init__(
        self,
        profile: DatasetProfile,
        *,
        input_key: str | None = None,
        target_key: str | None = None,
        coordinate_key: str | None = None,
        mask_key: str | None = None,
        input_fields: Sequence[str] | None = None,
        target_fields: Sequence[str] | None = None,
    ) -> None:
        self.profile = profile
        self.input_key = input_key
        self.target_key = target_key
        self.coordinate_key = coordinate_key
        self.mask_key = mask_key
        self.input_fields = tuple(input_fields or ())
        self.target_fields = tuple(target_fields or ())

    def _select(self, record: Mapping[str, Any], explicit: str | None, aliases: Sequence[str]) -> Any | None:
        if explicit is not None:
            if explicit not in record:
                raise DatasetAdapterError(f"Record does not contain required key {explicit!r}")
            return record[explicit]
        return _lookup(record, aliases)

    def adapt(self, record: Mapping[str, Any], *, index: int | None = None) -> CFDSample:
        raw_inputs = self._select(record, self.input_key, self.profile.input_aliases)
        raw_targets = self._select(record, self.target_key, self.profile.target_aliases)
        raw_coords = self._select(record, self.coordinate_key, self.profile.coordinate_aliases)
        raw_mask = self._select(record, self.mask_key, self.profile.mask_aliases)

        if raw_targets is None:
            raise DatasetAdapterError(
                f"Could not identify targets for {self.profile.dataset_id}; provide target_key or target_fields"
            )
        if raw_inputs is None:
            if raw_coords is None:
                raise DatasetAdapterError(
                    f"Could not identify inputs for {self.profile.dataset_id}; provide input_key or input_fields"
                )
            raw_inputs = raw_coords

        inputs = _combine_fields(raw_inputs, self.input_fields or None)
        targets = _combine_fields(raw_targets, self.target_fields or None)
        coordinates = _combine_fields(raw_coords) if raw_coords is not None else None
        mask = _to_numpy(raw_mask).astype(bool) if raw_mask is not None else None

        raw_parameters = _lookup(record, self.profile.parameter_aliases)
        parameters: MutableMapping[str, Any] = {}
        if isinstance(raw_parameters, Mapping):
            parameters.update(raw_parameters)
        elif raw_parameters is not None:
            parameters["value"] = raw_parameters

        reserved = {
            self.input_key,
            self.target_key,
            self.coordinate_key,
            self.mask_key,
            *self.profile.input_aliases,
            *self.profile.target_aliases,
            *self.profile.coordinate_aliases,
            *self.profile.mask_aliases,
            *self.profile.parameter_aliases,
        }
        metadata = {
            key: value
            for key, value in record.items()
            if key not in reserved and isinstance(value, (str, int, float, bool, type(None)))
        }
        metadata.update(
            {
                "dataset_id": self.profile.dataset_id,
                "representation": self.profile.representation,
            }
        )
        if index is not None:
            metadata["index"] = index

        return CFDSample(
            inputs=_to_numpy(inputs).astype(np.float32, copy=False),
            targets=_to_numpy(targets).astype(np.float32, copy=False),
            coordinates=(
                _to_numpy(coordinates).astype(np.float32, copy=False) if coordinates is not None else None
            ),
            parameters=parameters,
            mask=mask,
            metadata=metadata,
        )


class AdapterRegistry:
    def __init__(self, profiles: Mapping[str, DatasetProfile] | None = None) -> None:
        self._profiles = dict(profiles or BUILTIN_DATASET_PROFILES)

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def profile(self, dataset_id: str) -> DatasetProfile:
        try:
            return self._profiles[dataset_id]
        except KeyError as exc:
            raise KeyError(f"No dataset adapter profile registered for {dataset_id!r}") from exc

    def adapter(self, dataset_id: str, **kwargs: Any) -> DatasetAdapter:
        return DatasetAdapter(self.profile(dataset_id), **kwargs)

    def register(self, profile: DatasetProfile, *, replace: bool = False) -> None:
        if profile.dataset_id in self._profiles and not replace:
            raise ValueError(f"Dataset profile {profile.dataset_id!r} is already registered")
        self._profiles[profile.dataset_id] = profile


class AdaptedDataset:
    def __init__(self, raw_dataset: Any, adapter: DatasetAdapter) -> None:
        self.raw_dataset = raw_dataset
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> CFDSample:
        record = self.raw_dataset[index]
        if not isinstance(record, Mapping):
            raise DatasetAdapterError("Raw dataset items must be mapping-like records")
        return self.adapter.adapt(record, index=index)


class DatasetSubset:
    def __init__(self, dataset: Any, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]


def split_indices(
    length: int,
    *,
    train: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
    seed: int = 0,
) -> dict[str, tuple[int, ...]]:
    if length < 1:
        raise ValueError("length must be positive")
    fractions = (train, validation, test)
    if any(value < 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("train, validation and test fractions must be non-negative and sum to 1")
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    n_train = int(round(length * train))
    n_validation = int(round(length * validation))
    if n_train + n_validation > length:
        n_validation = length - n_train
    return {
        "train": tuple(indices[:n_train]),
        "validation": tuple(indices[n_train : n_train + n_validation]),
        "test": tuple(indices[n_train + n_validation :]),
    }


def split_dataset(dataset: Any, **kwargs: Any) -> dict[str, DatasetSubset]:
    return {
        name: DatasetSubset(dataset, indices)
        for name, indices in split_indices(len(dataset), **kwargs).items()
    }


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Data loaders require PyTorch; install `navier-cfd[models]`") from exc
    return torch


def _pad_first_axis(arrays: Sequence[Any], *, value: float = 0.0) -> tuple[Any, Any]:
    torch = _require_torch()
    tensors = [torch.as_tensor(array) for array in arrays]
    if all(tensor.shape == tensors[0].shape for tensor in tensors):
        stacked = torch.stack(tensors)
        mask = torch.ones((len(tensors), tensors[0].shape[0]), dtype=torch.bool)
        return stacked, mask
    if any(tensor.ndim == 0 for tensor in tensors):
        raise DatasetAdapterError("Cannot pad scalar arrays")
    trailing = tensors[0].shape[1:]
    if any(tensor.shape[1:] != trailing for tensor in tensors):
        raise DatasetAdapterError("Variable-size batches may differ only along their first axis")
    maximum = max(tensor.shape[0] for tensor in tensors)
    result = torch.full((len(tensors), maximum, *trailing), value, dtype=tensors[0].dtype)
    mask = torch.zeros((len(tensors), maximum), dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        result[index, : tensor.shape[0]] = tensor
        mask[index, : tensor.shape[0]] = True
    return result, mask


def collate_cfd_samples(samples: Sequence[CFDSample]) -> CFDBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    torch = _require_torch()
    inputs, inferred_mask = _pad_first_axis([sample.inputs for sample in samples])
    targets, target_mask = _pad_first_axis([sample.targets for sample in samples])
    if inputs.shape == targets.shape and inputs.ndim > 3:
        mask = torch.ones(inputs.shape[:-1], dtype=torch.bool)
    else:
        mask = inferred_mask & target_mask

    coordinates = None
    if all(sample.coordinates is not None for sample in samples):
        coordinates, coordinate_mask = _pad_first_axis([sample.coordinates for sample in samples])
        if coordinates.ndim <= 3:
            mask = mask & coordinate_mask

    explicit_masks = [sample.mask for sample in samples]
    if all(item is not None for item in explicit_masks):
        padded_masks, _ = _pad_first_axis(explicit_masks)
        if padded_masks.ndim > 2:
            padded_masks = padded_masks.reshape(padded_masks.shape[0], padded_masks.shape[1], -1).all(-1)
        mask = mask & padded_masks.bool()

    parameter_keys = sorted(set().union(*(sample.parameters.keys() for sample in samples)))
    parameters: dict[str, Any] = {}
    for key in parameter_keys:
        values = [sample.parameters.get(key, np.nan) for sample in samples]
        try:
            parameters[key] = torch.as_tensor(values)
        except (TypeError, ValueError):
            parameters[key] = tuple(values)

    return CFDBatch(
        inputs=inputs.float(),
        targets=targets.float(),
        coordinates=coordinates.float() if coordinates is not None else None,
        parameters=parameters,
        mask=mask,
        metadata=tuple(sample.metadata for sample in samples),
    )


def make_dataloaders(
    dataset: Any,
    *,
    batch_size: int = 8,
    train: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, Any]:
    torch = _require_torch()
    subsets = split_dataset(
        dataset,
        train=train,
        validation=validation,
        test=test,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_cfd_samples,
        )
        for name, subset in subsets.items()
    }


__all__ = [
    "AdaptedDataset",
    "AdapterRegistry",
    "BUILTIN_DATASET_PROFILES",
    "CFDBatch",
    "CFDSample",
    "DatasetAdapter",
    "DatasetAdapterError",
    "DatasetProfile",
    "DatasetSubset",
    "collate_cfd_samples",
    "make_dataloaders",
    "split_dataset",
    "split_indices",
]
