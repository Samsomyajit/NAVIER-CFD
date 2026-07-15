from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class CaseError:
    case_id: str
    rmse: float
    mae: float
    max_abs_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _broadcast_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    candidate = np.asarray(mask, dtype=bool)
    while candidate.ndim < len(shape):
        candidate = np.expand_dims(candidate, axis=-1)
    return np.broadcast_to(candidate, shape)


def _masked_flat(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if mask is not None:
        finite &= _broadcast_mask(np.asarray(mask, dtype=bool), array.shape)
    return array[finite]


def field_summary(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    selected = _masked_flat(array, mask)
    if selected.size == 0:
        raise ValueError("No finite values remain after applying the mask")
    return {
        "count": int(selected.size),
        "finite_fraction": float(np.isfinite(array).mean()),
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
        "rms": float(np.sqrt(np.mean(selected**2))),
    }


def conditioned_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    condition_mask: np.ndarray,
) -> float:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} != {target.shape}")
    mask = _broadcast_mask(np.asarray(condition_mask, dtype=bool), prediction.shape)
    valid = mask & np.isfinite(prediction) & np.isfinite(target)
    if not np.any(valid):
        raise ValueError("Condition mask selects no finite values")
    error = prediction[valid] - target[valid]
    return float(np.sqrt(np.mean(error**2)))


def gradient_interface_mask(
    phase_fraction: np.ndarray,
    *,
    percentile: float = 85.0,
    spatial_axes: Sequence[int] | None = None,
) -> np.ndarray:
    """Build a deterministic high-gradient interface mask.

    The mask is intended as a diagnostic proxy, not a claim that every selected cell is a
    physical phase interface. For batched arrays, pass the spatial axes explicitly.
    """

    alpha = np.asarray(phase_fraction, dtype=float)
    if alpha.ndim < 1:
        raise ValueError("phase_fraction must have at least one dimension")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    axes = tuple(spatial_axes) if spatial_axes is not None else tuple(range(max(0, alpha.ndim - 2), alpha.ndim))
    if not axes:
        axes = (alpha.ndim - 1,)
    magnitude_squared = np.zeros_like(alpha, dtype=float)
    for axis in axes:
        gradient = np.gradient(alpha, axis=axis)
        magnitude_squared += np.asarray(gradient, dtype=float) ** 2
    magnitude = np.sqrt(magnitude_squared)
    finite = np.isfinite(magnitude)
    positive = finite & (magnitude > 0)
    if not np.any(positive):
        raise ValueError("Cannot construct an interface mask because all gradients are zero")
    threshold = float(np.percentile(magnitude[positive], percentile))
    return positive & (magnitude >= threshold)


def analyze_interface_error(
    prediction: np.ndarray,
    target: np.ndarray,
    phase_fraction: np.ndarray,
    *,
    percentile: float = 85.0,
    spatial_axes: Sequence[int] | None = None,
) -> dict[str, Any]:
    mask = gradient_interface_mask(
        phase_fraction,
        percentile=percentile,
        spatial_axes=spatial_axes,
    )
    interface = conditioned_rmse(prediction, target, mask)
    bulk_mask = ~mask
    bulk = conditioned_rmse(prediction, target, bulk_mask)
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    valid = np.isfinite(error)
    total_energy = float(np.sum(error[valid] ** 2))
    interface_energy = float(np.sum(error[_broadcast_mask(mask, error.shape) & valid] ** 2))
    return {
        "interface_percentile": percentile,
        "interface_fraction": float(np.mean(mask)),
        "interface_rmse": interface,
        "bulk_rmse": bulk,
        "interface_to_bulk_rmse_ratio": interface / bulk if bulk > 0 else float("inf"),
        "interface_squared_error_fraction": interface_energy / total_energy if total_energy > 0 else 0.0,
    }


def rank_worst_cases(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    case_ids: Sequence[str] | None = None,
    mask: np.ndarray | None = None,
    top_k: int = 5,
) -> list[CaseError]:
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} != {targets.shape}")
    if predictions.ndim < 1:
        raise ValueError("Predictions must include a case axis")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    n_cases = predictions.shape[0]
    ids = list(case_ids) if case_ids is not None else [f"case_{index}" for index in range(n_cases)]
    if len(ids) != n_cases:
        raise ValueError("case_ids length must match the first array dimension")

    rows: list[CaseError] = []
    for index, case_id in enumerate(ids):
        prediction = predictions[index]
        target = targets[index]
        valid = np.isfinite(prediction) & np.isfinite(target)
        if mask is not None:
            local_mask = np.asarray(mask[index] if np.asarray(mask).shape == predictions.shape else mask, dtype=bool)
            valid &= _broadcast_mask(local_mask, prediction.shape)
        if not np.any(valid):
            continue
        error = prediction[valid] - target[valid]
        rows.append(
            CaseError(
                case_id=str(case_id),
                rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(np.abs(error))),
                max_abs_error=float(np.max(np.abs(error))),
            )
        )
    return sorted(rows, key=lambda row: row.rmse, reverse=True)[:top_k]
