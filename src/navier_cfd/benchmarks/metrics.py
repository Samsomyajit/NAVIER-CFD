from __future__ import annotations

from typing import Any

import numpy as np

from ..metrics import (
    MetricContext,
    cosine_similarity as _cosine_similarity,
    divergence_rms as _divergence_rms,
    kinetic_energy_relative_error,
    mae as _mae,
    r2 as _r2,
    relative_l2 as _relative_l2,
    rmse as _rmse,
    spectral_relative_error as _spectral_relative_error,
)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return _rmse(pred, target)


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return _mae(pred, target)


def normalized_rmse(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    scale = np.sqrt(np.mean((target - np.mean(target)) ** 2))
    return float(rmse(pred, target) / (scale + eps))


def relative_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    return _relative_l2(pred, target, eps=eps)


def r2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    return _r2(pred, target, eps=eps)


def cosine_similarity(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    return _cosine_similarity(pred, target, eps=eps)


def spectral_relative_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    return _spectral_relative_error(pred, target, eps=eps)


def divergence_rms(velocity: np.ndarray, spacing: tuple[float, ...] | None = None) -> float:
    ncomp = int(np.asarray(velocity).shape[-1])
    return _divergence_rms(
        velocity,
        MetricContext(
            velocity_channels=tuple(range(ncomp)),
            spacing=spacing,
            sample_axis=0 if np.asarray(velocity).ndim > ncomp + 1 else None,
        ),
    )


def kinetic_energy_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    ncomp = int(np.asarray(pred).shape[-1])
    return kinetic_energy_relative_error(
        pred,
        target,
        MetricContext(velocity_channels=tuple(range(ncomp))),
        eps=eps,
    )


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
