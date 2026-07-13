from __future__ import annotations

from typing import Any

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def normalized_rmse(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    scale = np.sqrt(np.mean((target - np.mean(target)) ** 2))
    return float(rmse(pred, target) / (scale + eps))


def relative_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(pred - target) / (np.linalg.norm(target) + eps))


def r2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    residual = np.sum((target - pred) ** 2)
    total = np.sum((target - np.mean(target)) ** 2)
    return float(1.0 - residual / (total + eps))


def cosine_similarity(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    p = pred.reshape(-1)
    t = target.reshape(-1)
    return float(np.dot(p, t) / ((np.linalg.norm(p) * np.linalg.norm(t)) + eps))


def spectral_relative_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    if pred.ndim < 2:
        return relative_l2(pred, target, eps)
    axes = tuple(range(-min(3, pred.ndim - 1), -1)) if pred.ndim > 2 else (-1,)
    p = np.abs(np.fft.rfftn(pred, axes=axes))
    t = np.abs(np.fft.rfftn(target, axes=axes))
    return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + eps))


def divergence_rms(velocity: np.ndarray, spacing: tuple[float, ...] | None = None) -> float:
    """Approximate RMS divergence for structured arrays shaped (..., spatial..., components)."""
    velocity = np.asarray(velocity)
    if velocity.shape[-1] not in {2, 3}:
        raise ValueError("velocity must store 2 or 3 components in the final axis")
    ncomp = velocity.shape[-1]
    spacing = spacing or tuple(1.0 for _ in range(ncomp))
    if len(spacing) != ncomp:
        raise ValueError("spacing length must equal the number of velocity components")
    if velocity.ndim < ncomp + 1:
        raise ValueError("velocity array does not contain enough spatial dimensions")
    div: Any = 0.0
    first_spatial_axis = velocity.ndim - ncomp - 1
    for component in range(ncomp):
        axis = first_spatial_axis + component
        div = div + np.gradient(velocity[..., component], spacing[component], axis=axis)
    return float(np.sqrt(np.mean(np.asarray(div) ** 2)))


def kinetic_energy_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    if pred.shape[-1] not in {2, 3}:
        raise ValueError("kinetic energy requires 2D or 3D velocity components")
    predicted_energy = 0.5 * np.sum(pred**2, axis=-1)
    target_energy = 0.5 * np.sum(target**2, axis=-1)
    return float(np.linalg.norm(predicted_energy - target_energy) / (np.linalg.norm(target_energy) + eps))


def rollout_error_curve(pred: np.ndarray, target: np.ndarray, time_axis: int = 1) -> list[float]:
    pred = np.moveaxis(np.asarray(pred), time_axis, 0)
    target = np.moveaxis(np.asarray(target), time_axis, 0)
    if pred.shape != target.shape:
        raise ValueError("rollout arrays must have matching shapes")
    return [relative_l2(pred[index], target[index]) for index in range(pred.shape[0])]


def compute_metric_bundle(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    velocity: bool = False,
    spacing: tuple[float, ...] | None = None,
    rollout_time_axis: int | None = None,
) -> dict[str, Any]:
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} != {target.shape}")
    metrics: dict[str, Any] = {
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "normalized_rmse": normalized_rmse(pred, target),
        "relative_l2": relative_l2(pred, target),
        "r2": r2(pred, target),
        "cosine_similarity": cosine_similarity(pred, target),
        "spectral_relative_error": spectral_relative_error(pred, target),
        "max_absolute_error": float(np.max(np.abs(pred - target))),
    }
    if velocity:
        metrics["divergence_rms_pred"] = divergence_rms(pred, spacing)
        metrics["divergence_rms_target"] = divergence_rms(target, spacing)
        metrics["kinetic_energy_relative_error"] = kinetic_energy_error(pred, target)
    if rollout_time_axis is not None:
        curve = rollout_error_curve(pred, target, rollout_time_axis)
        metrics["rollout_relative_l2"] = curve
        metrics["rollout_final_relative_l2"] = curve[-1]
        metrics["rollout_mean_relative_l2"] = float(np.mean(curve))
    return metrics


__all__ = [
    "compute_metric_bundle",
    "cosine_similarity",
    "divergence_rms",
    "kinetic_energy_error",
    "mae",
    "normalized_rmse",
    "r2",
    "relative_l2",
    "rmse",
    "rollout_error_curve",
    "spectral_relative_error",
]
