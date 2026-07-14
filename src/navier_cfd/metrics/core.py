from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MetricContext:
    """Physical and tensor metadata required by evaluation metrics."""

    mask: Any | None = None
    spatial_axes: tuple[int, ...] | None = None
    time_axis: int | None = None
    channel_axis: int = -1
    sample_axis: int | None = 0
    spacing: tuple[float, ...] | None = None
    velocity_channels: tuple[int, ...] | None = None
    density_channel: int | None = None
    profile_axis: int | None = None
    probe_indices: tuple[tuple[int, ...], ...] = ()
    field_variance: Any | None = None
    evaluation_space: str = "physical"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: Any
    category: str
    direction: str
    best_value: float | None
    units: str = "dimensionless"
    valid: bool = True
    reason: str | None = None
    definition: str = ""
    assumptions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        value = payload["value"]
        if isinstance(value, np.ndarray):
            payload["value"] = value.tolist()
        elif isinstance(value, np.generic):
            payload["value"] = value.item()
        return payload


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    function: Any
    category: str = "data"
    direction: str = "lower"
    best_value: float | None = 0.0
    units: str = "dimensionless"
    definition: str = ""
    requires: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def evaluate(self, prediction: Any, target: Any, context: MetricContext) -> MetricResult:
        missing = [name for name in self.requires if getattr(context, name, None) in (None, ())]
        if missing:
            return MetricResult(
                name=self.name,
                value=float("nan"),
                category=self.category,
                direction=self.direction,
                best_value=self.best_value,
                units=self.units,
                valid=False,
                reason=f"missing metric context: {', '.join(missing)}",
                definition=self.definition,
                assumptions=self.assumptions,
                metadata={"evaluation_space": context.evaluation_space},
            )
        try:
            value = self.function(prediction, target, context)
        except (ValueError, IndexError, TypeError) as exc:
            return MetricResult(
                name=self.name,
                value=float("nan"),
                category=self.category,
                direction=self.direction,
                best_value=self.best_value,
                units=self.units,
                valid=False,
                reason=str(exc),
                definition=self.definition,
                assumptions=self.assumptions,
                metadata={"evaluation_space": context.evaluation_space},
            )
        return MetricResult(
            name=self.name,
            value=value,
            category=self.category,
            direction=self.direction,
            best_value=self.best_value,
            units=self.units,
            definition=self.definition,
            assumptions=self.assumptions,
            metadata={"evaluation_space": context.evaluation_space},
        )


def as_arrays(prediction: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction)
    truth = np.asarray(target)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {pred.shape} != {truth.shape}")
    if pred.size == 0:
        raise ValueError("metric arrays must be non-empty")
    return pred, truth


def masked_flatten(
    prediction: Any,
    target: Any,
    mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pred, truth = as_arrays(prediction, target)
    if mask is None:
        return pred.reshape(-1), truth.reshape(-1)
    selected = np.asarray(mask, dtype=bool)
    while selected.ndim < pred.ndim:
        selected = np.expand_dims(selected, axis=-1)
    try:
        selected = np.broadcast_to(selected, pred.shape)
    except ValueError as exc:
        raise ValueError(f"mask shape {np.asarray(mask).shape} cannot broadcast to {pred.shape}") from exc
    return pred[selected], truth[selected]


def normalize_axis(axis: int, ndim: int) -> int:
    return axis if axis >= 0 else ndim + axis


def infer_spatial_axes(array: np.ndarray, context: MetricContext) -> tuple[int, ...]:
    if context.spatial_axes is not None:
        return tuple(normalize_axis(axis, array.ndim) for axis in context.spatial_axes)
    excluded = {normalize_axis(context.channel_axis, array.ndim)}
    if context.sample_axis is not None:
        excluded.add(normalize_axis(context.sample_axis, array.ndim))
    if context.time_axis is not None:
        excluded.add(normalize_axis(context.time_axis, array.ndim))
    return tuple(axis for axis in range(array.ndim) if axis not in excluded)


def select_channels(array: np.ndarray, channels: Sequence[int], channel_axis: int = -1) -> np.ndarray:
    return np.take(array, tuple(channels), axis=channel_axis)


__all__ = [
    "MetricContext",
    "MetricDefinition",
    "MetricResult",
    "as_arrays",
    "infer_spatial_axes",
    "masked_flatten",
    "normalize_axis",
    "select_channels",
]
