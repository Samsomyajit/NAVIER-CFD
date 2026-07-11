from __future__ import annotations

from typing import Any

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def relative_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(pred - target) / (np.linalg.norm(target) + eps))


def r2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    residual = np.sum((target - pred) ** 2)
    total = np.sum((target - np.mean(target)) ** 2)
    return float(1.0 - residual / (total + eps))


def spectral_relative_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    axes = tuple(range(-min(2, pred.ndim), 0))
    p = np.abs(np.fft.rfftn(pred, axes=axes))
    t = np.abs(np.fft.rfftn(target, axes=axes))
    return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + eps))


def divergence_rms(velocity: np.ndarray, spacing: tuple[float, ...] | None = None) -> float:
    """Approximate RMS divergence for arrays shaped (..., spatial..., components)."""
    if velocity.shape[-1] not in {2, 3}:
        raise ValueError("velocity must store 2 or 3 components in the final axis")
    ncomp = velocity.shape[-1]
    spacing = spacing or tuple(1.0 for _ in range(ncomp))
    if len(spacing) != ncomp:
        raise ValueError("spacing length must equal the number of velocity components")
    div = 0.0
    for component in range(ncomp):
        axis = velocity.ndim - ncomp - 1 + component
        div = div + np.gradient(velocity[..., component], spacing[component], axis=axis)
    return float(np.sqrt(np.mean(np.asarray(div) ** 2)))


def compute_metric_bundle(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    velocity: bool = False,
    spacing: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} != {target.shape}")
    metrics: dict[str, Any] = {
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "relative_l2": relative_l2(pred, target),
        "r2": r2(pred, target),
        "spectral_relative_error": spectral_relative_error(pred, target),
    }
    if velocity:
        metrics["divergence_rms_pred"] = divergence_rms(pred, spacing)
        metrics["divergence_rms_target"] = divergence_rms(target, spacing)
    return metrics
