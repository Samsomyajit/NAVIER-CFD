from __future__ import annotations

from typing import Any

import numpy as np

from .core import MetricContext, as_arrays, masked_flatten, normalize_axis


def mse(prediction: Any, target: Any, context: MetricContext | None = None) -> float:
    context = context or MetricContext()
    pred, truth = masked_flatten(prediction, target, context.mask)
    return float(np.mean((pred - truth) ** 2))


def rmse(prediction: Any, target: Any, context: MetricContext | None = None) -> float:
    return float(np.sqrt(mse(prediction, target, context)))


def mae(prediction: Any, target: Any, context: MetricContext | None = None) -> float:
    context = context or MetricContext()
    pred, truth = masked_flatten(prediction, target, context.mask)
    return float(np.mean(np.abs(pred - truth)))


def linfinity(prediction: Any, target: Any, context: MetricContext | None = None) -> float:
    context = context or MetricContext()
    pred, truth = masked_flatten(prediction, target, context.mask)
    return float(np.max(np.abs(pred - truth)))


def relative_l1(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    eps: float = 1e-12,
) -> float:
    context = context or MetricContext()
    pred, truth = masked_flatten(prediction, target, context.mask)
    return float(np.sum(np.abs(pred - truth)) / (np.sum(np.abs(truth)) + eps))


def relative_l2(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    eps: float = 1e-12,
    per_sample: bool = False,
) -> float:
    context = context or MetricContext()
    pred, truth = as_arrays(prediction, target)
    if not per_sample or context.sample_axis is None:
        p, t = masked_flatten(pred, truth, context.mask)
        return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + eps))

    sample_axis = normalize_axis(context.sample_axis, pred.ndim)
    pred = np.moveaxis(pred, sample_axis, 0)
    truth = np.moveaxis(truth, sample_axis, 0)
    if context.mask is not None:
        mask = np.asarray(context.mask)
        if mask.ndim == pred.ndim:
            mask = np.moveaxis(mask, sample_axis, 0)
        else:
            mask = np.broadcast_to(mask, pred.shape)
            mask = np.moveaxis(mask, sample_axis, 0)
    else:
        mask = None
    values = []
    for index in range(pred.shape[0]):
        selected = mask[index] if mask is not None else None
        p, t = masked_flatten(pred[index], truth[index], selected)
        values.append(np.linalg.norm(p - t) / (np.linalg.norm(t) + eps))
    return float(np.mean(values))


def nmse(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    eps: float = 1e-12,
) -> float:
    context = context or MetricContext()
    pred, truth = masked_flatten(prediction, target, context.mask)
    return float(np.mean((pred - truth) ** 2) / (np.mean(truth**2) + eps))


def nrmse(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    eps: float = 1e-12,
) -> float:
    return float(np.sqrt(nmse(prediction, target, context, eps)))


def r2(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    eps: float = 1e-12,
) -> float:
    context = context or MetricContext()
    pred, truth = masked_flatten(prediction, target, context.mask)
    residual = np.sum((truth - pred) ** 2)
    total = np.sum((truth - np.mean(truth)) ** 2)
    return float(1.0 - residual / (total + eps))


def pearson_r(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    eps: float = 1e-12,
) -> float:
    context = context or MetricContext()
    pred, truth = masked_flatten(prediction, target, context.mask)
    pred = pred - np.mean(pred)
    truth = truth - np.mean(truth)
    return float(np.dot(pred, truth) / (np.linalg.norm(pred) * np.linalg.norm(truth) + eps))


def cosine_similarity(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    eps: float = 1e-12,
) -> float:
    context = context or MetricContext()
    pred, truth = masked_flatten(prediction, target, context.mask)
    return float(np.dot(pred, truth) / (np.linalg.norm(pred) * np.linalg.norm(truth) + eps))


def _channel_variance(target: np.ndarray, context: MetricContext, eps: float) -> np.ndarray:
    channel_axis = normalize_axis(context.channel_axis, target.ndim)
    if context.field_variance is not None:
        variance = np.asarray(context.field_variance, dtype=np.float64)
    else:
        axes = tuple(axis for axis in range(target.ndim) if axis != channel_axis)
        variance = np.var(target, axis=axes)
    return np.maximum(variance, eps)


def variance_scaled_mse(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    eps: float = 1e-12,
) -> float:
    context = context or MetricContext()
    pred, truth = as_arrays(prediction, target)
    channel_axis = normalize_axis(context.channel_axis, truth.ndim)
    axes = tuple(axis for axis in range(truth.ndim) if axis != channel_axis)
    per_channel_mse = np.mean((pred - truth) ** 2, axis=axes)
    return float(np.mean(per_channel_mse / _channel_variance(truth, context, eps)))


def variance_scaled_rmse(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    eps: float = 1e-12,
) -> float:
    return float(np.sqrt(variance_scaled_mse(prediction, target, context, eps)))


def update_ratio(finetuning_updates: int, scratch_updates: int) -> float:
    if finetuning_updates < 0 or scratch_updates <= 0:
        raise ValueError("finetuning_updates must be non-negative and scratch_updates positive")
    return float(finetuning_updates / scratch_updates)


__all__ = [
    "cosine_similarity",
    "linfinity",
    "mae",
    "mse",
    "nmse",
    "nrmse",
    "pearson_r",
    "r2",
    "relative_l1",
    "relative_l2",
    "rmse",
    "update_ratio",
    "variance_scaled_mse",
    "variance_scaled_rmse",
]
