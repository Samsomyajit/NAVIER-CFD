from __future__ import annotations

from typing import Any

import numpy as np

from .core import MetricContext, as_arrays, infer_spatial_axes, normalize_axis, select_channels


def divergence_rms(
    velocity: Any,
    context: MetricContext,
) -> float:
    array = np.asarray(velocity)
    channels = context.velocity_channels
    if not channels:
        raise ValueError("divergence requires velocity_channels")
    values = select_channels(array, channels, context.channel_axis)
    spatial_axes = infer_spatial_axes(values, context)
    if len(spatial_axes) < len(channels):
        raise ValueError("not enough spatial axes for velocity components")
    spacing = context.spacing or tuple(1.0 for _ in channels)
    if len(spacing) != len(channels):
        raise ValueError("spacing length must equal number of velocity components")
    divergence: Any = 0.0
    for component, axis in enumerate(spatial_axes[: len(channels)]):
        component_values = np.take(values, component, axis=context.channel_axis)
        divergence = divergence + np.gradient(component_values, spacing[component], axis=axis)
    return float(np.sqrt(np.mean(np.asarray(divergence) ** 2)))


def divergence_error(
    prediction: Any,
    target: Any,
    context: MetricContext,
) -> float:
    return float(abs(divergence_rms(prediction, context) - divergence_rms(target, context)))


def kinetic_energy_relative_error(
    prediction: Any,
    target: Any,
    context: MetricContext,
    eps: float = 1e-12,
) -> float:
    pred, truth = as_arrays(prediction, target)
    channels = context.velocity_channels
    if not channels:
        raise ValueError("kinetic energy requires velocity_channels")
    pred_velocity = select_channels(pred, channels, context.channel_axis)
    truth_velocity = select_channels(truth, channels, context.channel_axis)
    pred_energy = 0.5 * np.sum(pred_velocity**2, axis=context.channel_axis)
    truth_energy = 0.5 * np.sum(truth_velocity**2, axis=context.channel_axis)
    if context.density_channel is not None:
        pred_density = np.take(pred, context.density_channel, axis=context.channel_axis)
        truth_density = np.take(truth, context.density_channel, axis=context.channel_axis)
        pred_energy = pred_density * pred_energy
        truth_energy = truth_density * truth_energy
    return float(np.linalg.norm(pred_energy - truth_energy) / (np.linalg.norm(truth_energy) + eps))


def turbulent_kinetic_energy_error(
    prediction: Any,
    target: Any,
    context: MetricContext,
) -> float:
    """RealPDEBench-style error in fluctuation kinetic energy."""

    pred, truth = as_arrays(prediction, target)
    if context.time_axis is None:
        raise ValueError("turbulent kinetic energy requires time_axis")
    channels = context.velocity_channels
    if not channels or len(channels) < 2:
        raise ValueError("turbulent kinetic energy requires at least two velocity channels")
    time_axis = normalize_axis(context.time_axis, pred.ndim)
    pred_velocity = select_channels(pred, channels, context.channel_axis)
    truth_velocity = select_channels(truth, channels, context.channel_axis)
    pred_mean = np.mean(pred_velocity, axis=time_axis, keepdims=True)
    truth_mean = np.mean(truth_velocity, axis=time_axis, keepdims=True)
    pred_tke = 0.5 * np.mean((pred_velocity - pred_mean) ** 2, axis=(time_axis, context.channel_axis))
    truth_tke = 0.5 * np.mean((truth_velocity - truth_mean) ** 2, axis=(time_axis, context.channel_axis))
    return float(np.mean(np.abs(pred_tke - truth_tke)))


def mean_velocity_profile_error(
    prediction: Any,
    target: Any,
    context: MetricContext,
) -> float:
    """Compare time-averaged velocity values at probes or along a profile axis."""

    pred, truth = as_arrays(prediction, target)
    if context.time_axis is None:
        raise ValueError("mean velocity profile error requires time_axis")
    channels = context.velocity_channels
    if not channels:
        raise ValueError("mean velocity profile error requires velocity_channels")
    time_axis = normalize_axis(context.time_axis, pred.ndim)
    pred_velocity = select_channels(pred, channels, context.channel_axis)
    truth_velocity = select_channels(truth, channels, context.channel_axis)
    pred_mean = np.mean(pred_velocity, axis=time_axis)
    truth_mean = np.mean(truth_velocity, axis=time_axis)

    if context.probe_indices:
        errors = []
        sample_axis = context.sample_axis
        for probe in context.probe_indices:
            if sample_axis is None:
                index = tuple(probe) + (slice(None),)
            else:
                index = (slice(None),) + tuple(probe) + (slice(None),)
            errors.append(np.mean(np.abs(pred_mean[index] - truth_mean[index])))
        return float(np.mean(errors))

    if context.profile_axis is None:
        raise ValueError("provide probe_indices or profile_axis")
    profile_axis = normalize_axis(context.profile_axis, pred.ndim)
    if profile_axis > time_axis:
        profile_axis -= 1
    channel_axis = normalize_axis(context.channel_axis, pred.ndim)
    if channel_axis > time_axis:
        channel_axis -= 1
    keep = {profile_axis, channel_axis}
    if context.sample_axis is not None:
        sample_axis = normalize_axis(context.sample_axis, pred.ndim)
        if sample_axis > time_axis:
            sample_axis -= 1
        keep.add(sample_axis)
    reduction_axes = tuple(axis for axis in range(pred_mean.ndim) if axis not in keep)
    if reduction_axes:
        pred_profile = np.mean(pred_mean, axis=reduction_axes)
        truth_profile = np.mean(truth_mean, axis=reduction_axes)
    else:
        pred_profile, truth_profile = pred_mean, truth_mean
    return float(np.mean(np.abs(pred_profile - truth_profile)))


def vorticity_rmse(
    prediction: Any,
    target: Any,
    context: MetricContext,
) -> float:
    pred, truth = as_arrays(prediction, target)
    channels = context.velocity_channels
    if not channels or len(channels) != 2:
        raise ValueError("2D vorticity requires exactly two velocity channels")
    spatial_axes = infer_spatial_axes(pred, context)
    if len(spatial_axes) < 2:
        raise ValueError("2D vorticity requires two spatial axes")
    spacing = context.spacing or (1.0, 1.0)

    def curl(array: np.ndarray) -> np.ndarray:
        u = np.take(array, channels[0], axis=context.channel_axis)
        v = np.take(array, channels[1], axis=context.channel_axis)
        dv_dx = np.gradient(v, spacing[0], axis=spatial_axes[0])
        du_dy = np.gradient(u, spacing[1], axis=spatial_axes[1])
        return dv_dx - du_dy

    return float(np.sqrt(np.mean((curl(pred) - curl(truth)) ** 2)))


__all__ = [
    "divergence_error",
    "divergence_rms",
    "kinetic_energy_relative_error",
    "mean_velocity_profile_error",
    "turbulent_kinetic_energy_error",
    "vorticity_rmse",
]
