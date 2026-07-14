from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .core import CFDBatch, CFDSample, DatasetAdapterError, split_dataset


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Data loaders require PyTorch; install `navier-cfd[models]`") from exc
    return torch


def _stack_or_pad(arrays: Sequence[Any], *, value: float = 0.0) -> tuple[Any, Any, bool]:
    torch = _require_torch()
    tensors = [torch.as_tensor(array) for array in arrays]
    if all(tensor.shape == tensors[0].shape for tensor in tensors):
        return torch.stack(tensors), torch.ones((len(tensors), tensors[0].shape[0]), dtype=torch.bool), False
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
    return result, mask, True


def collate_cfd_samples(samples: Sequence[CFDSample]) -> CFDBatch:
    """Collate equal structured grids or pad variable-size point/mesh samples."""

    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    torch = _require_torch()
    inputs, input_mask, inputs_padded = _stack_or_pad([sample.inputs for sample in samples])
    targets, target_mask, targets_padded = _stack_or_pad([sample.targets for sample in samples])

    same_spatial = inputs.ndim > 2 and inputs.shape[:-1] == targets.shape[:-1]
    if same_spatial and not inputs_padded and not targets_padded:
        mask = torch.ones(inputs.shape[:-1], dtype=torch.bool)
    else:
        mask = input_mask & target_mask

    coordinates = None
    if all(sample.coordinates is not None for sample in samples):
        coordinates, coordinate_mask, coordinates_padded = _stack_or_pad(
            [sample.coordinates for sample in samples]
        )
        if coordinates_padded or mask.ndim == 2:
            mask = mask & coordinate_mask

    explicit_masks = [sample.mask for sample in samples]
    if all(item is not None for item in explicit_masks):
        explicit, _, _ = _stack_or_pad(explicit_masks)
        explicit = explicit.bool()
        if explicit.shape == mask.shape:
            mask = mask & explicit
        elif explicit.ndim == mask.ndim + 1 and explicit.shape[-1] == 1:
            mask = mask & explicit.squeeze(-1)
        elif mask.ndim == 2:
            explicit = explicit.reshape(explicit.shape[0], explicit.shape[1], -1).all(-1)
            mask = mask & explicit
        else:
            raise DatasetAdapterError(
                f"Explicit mask shape {tuple(explicit.shape)} is incompatible with {tuple(mask.shape)}"
            )

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


def make_split_dataloaders(
    datasets: Mapping[str, Any],
    *,
    batch_size: int = 8,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> dict[str, Any]:
    """Create loaders from already separated provider-native datasets.

    Accepted validation keys are ``validation`` or ``valid``. No additional random
    splitting is performed, which prevents trajectory leakage across official splits.
    """

    torch = _require_torch()
    normalized = dict(datasets)
    if "validation" not in normalized and "valid" in normalized:
        normalized["validation"] = normalized.pop("valid")
    required = {"train", "validation", "test"}
    missing = required - set(normalized)
    if missing:
        raise ValueError(f"official split datasets are missing: {', '.join(sorted(missing))}")
    for name in required:
        if len(normalized[name]) < 1:
            raise ValueError(f"official {name} dataset is empty")

    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.utils.data.DataLoader(
            normalized[name],
            batch_size=batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last and name == "train",
            collate_fn=collate_cfd_samples,
        )
        for name in ("train", "validation", "test")
    }


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
    drop_last: bool = False,
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
            drop_last=drop_last and name == "train",
            collate_fn=collate_cfd_samples,
        )
        for name, subset in subsets.items()
    }


__all__ = ["collate_cfd_samples", "make_dataloaders", "make_split_dataloaders"]
