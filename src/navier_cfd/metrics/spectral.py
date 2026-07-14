from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .core import MetricContext, as_arrays, infer_spatial_axes, normalize_axis
from .data import relative_l2


def spectral_relative_error(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    eps: float = 1e-12,
) -> float:
    context = context or MetricContext()
    pred, truth = as_arrays(prediction, target)
    axes = infer_spatial_axes(pred, context)
    if not axes:
        return relative_l2(pred, truth, context, eps)
    pred_fft = np.fft.fftn(pred, axes=axes, norm="ortho")
    truth_fft = np.fft.fftn(truth, axes=axes, norm="ortho")
    return float(np.linalg.norm(np.abs(pred_fft) - np.abs(truth_fft)) / (np.linalg.norm(np.abs(truth_fft)) + eps))


def _radial_frequency_grid(shape: Sequence[int]) -> np.ndarray:
    components = np.meshgrid(
        *[2.0 * np.pi * np.fft.fftfreq(int(size)) for size in shape],
        indexing="ij",
    )
    radius = np.sqrt(sum(component**2 for component in components))
    maximum = float(np.max(radius))
    if maximum > 0:
        radius = np.pi * radius / maximum
    return radius


def binned_spectral_mse(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    bins: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute Parseval-consistent MSE contributions over radial wavenumber bins.

    Default bins partition the normalized radial spectrum into low, middle, and high
    bands. With orthonormal FFT normalization, the sum of bin contributions is equal
    to spatial MSE up to numerical precision when all Fourier modes are included.
    """

    context = context or MetricContext()
    pred, truth = as_arrays(prediction, target)
    axes = infer_spatial_axes(pred, context)
    if not axes:
        raise ValueError("binned spectral MSE requires at least one spatial axis")
    edges = np.asarray(bins or (0.0, np.pi / 4.0, np.pi / 2.0, np.inf), dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("bins must be a strictly increasing one-dimensional sequence")

    error_fft = np.fft.fftn(pred - truth, axes=axes, norm="ortho")
    power = np.abs(error_fft) ** 2
    spatial_shape = tuple(pred.shape[axis] for axis in axes)
    radial = _radial_frequency_grid(spatial_shape)
    radial_shape = [1] * pred.ndim
    for index, axis in enumerate(axes):
        radial_shape[axis] = spatial_shape[index]
    radial = radial.reshape(radial_shape)

    total_elements = float(pred.size)
    labels = ["low", "middle", "high"] if len(edges) == 4 else [f"bin_{i}" for i in range(len(edges) - 1)]
    values: dict[str, float] = {}
    for index, label in enumerate(labels):
        lower, upper = edges[index], edges[index + 1]
        selected = (radial >= lower) & (radial < upper)
        values[label] = float(np.sum(np.where(selected, power, 0.0)) / total_elements)
    values["total"] = float(sum(values.values()))
    return values


def fourier_rmse_bins(
    prediction: Any,
    target: Any,
    context: MetricContext | None = None,
    bins: Sequence[float] | None = None,
) -> dict[str, float]:
    """RealPDEBench-style Fourier RMSE over low/middle/high bands."""

    context = context or MetricContext()
    pred, truth = as_arrays(prediction, target)
    axes = list(infer_spatial_axes(pred, context))
    if context.time_axis is not None:
        time_axis = normalize_axis(context.time_axis, pred.ndim)
        if time_axis not in axes:
            axes.insert(0, time_axis)
    axes_tuple = tuple(axes)
    if not axes_tuple:
        raise ValueError("fRMSE requires temporal or spatial transform axes")
    edges = np.asarray(bins or (0.0, np.pi / 4.0, np.pi / 2.0, np.inf), dtype=float)
    error_fft = np.fft.fftn(pred - truth, axes=axes_tuple, norm="ortho")
    power = np.abs(error_fft) ** 2
    transformed_shape = tuple(pred.shape[axis] for axis in axes_tuple)
    radial = _radial_frequency_grid(transformed_shape)
    radial_shape = [1] * pred.ndim
    for index, axis in enumerate(axes_tuple):
        radial_shape[axis] = transformed_shape[index]
    radial = radial.reshape(radial_shape)
    labels = ["low", "middle", "high"] if len(edges) == 4 else [f"bin_{i}" for i in range(len(edges) - 1)]
    values: dict[str, float] = {}
    for index, label in enumerate(labels):
        selected = (radial >= edges[index]) & (radial < edges[index + 1])
        count = int(np.count_nonzero(np.broadcast_to(selected, power.shape)))
        values[label] = float(np.sqrt(np.sum(np.where(selected, power, 0.0)) / max(count, 1)))
    return values


def frequency_error(
    prediction: Any,
    target: Any,
    context: MetricContext,
) -> float:
    """Compare temporal spectra of spatially summed signals."""

    pred, truth = as_arrays(prediction, target)
    if context.time_axis is None:
        raise ValueError("frequency error requires time_axis")
    time_axis = normalize_axis(context.time_axis, pred.ndim)
    spatial_axes = infer_spatial_axes(pred, context)
    pred_signal = np.sum(pred, axis=spatial_axes) if spatial_axes else pred
    truth_signal = np.sum(truth, axis=spatial_axes) if spatial_axes else truth
    adjusted_time_axis = time_axis - sum(axis < time_axis for axis in spatial_axes)
    pred_fft = np.fft.fft(pred_signal, axis=adjusted_time_axis, norm="ortho")
    truth_fft = np.fft.fft(truth_signal, axis=adjusted_time_axis, norm="ortho")
    return float(np.mean(np.abs(pred_fft - truth_fft)))


__all__ = [
    "binned_spectral_mse",
    "fourier_rmse_bins",
    "frequency_error",
    "spectral_relative_error",
]
