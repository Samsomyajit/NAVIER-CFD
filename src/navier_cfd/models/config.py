from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..datasets.core import CFDSample


@dataclass(frozen=True)
class ModelBuildPlan:
    model_id: str
    builder_kwargs: Mapping[str, Any]
    input_mode: str
    notes: tuple[str, ...] = field(default_factory=tuple)


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError("Sample arrays must expose shape")
    return tuple(int(item) for item in shape)


def translate_model_config(
    model_id: str,
    sample: CFDSample,
    *,
    task: Any | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ModelBuildPlan:
    """Translate a canonical sample into model-constructor arguments.

    The translation is returned before model construction so every inferred
    dimension, channel count, and tensor mode can be reviewed and recorded.
    """

    input_shape = _shape(sample.inputs)
    target_shape = _shape(sample.targets)
    input_channels = input_shape[-1] if len(input_shape) > 1 else 1
    output_channels = target_shape[-1] if len(target_shape) > 1 else 1
    dimension = getattr(task, "dimension", None)
    if dimension is None:
        if sample.coordinates is not None:
            coordinate_shape = _shape(sample.coordinates)
            dimension = coordinate_shape[-1] if len(coordinate_shape) > 1 else 1
        else:
            dimension = max(1, len(input_shape) - 1)

    kwargs: dict[str, Any]
    mode: str
    notes: list[str] = []

    if model_id == "fno":
        kwargs = {
            "dimension": int(dimension),
            "in_channels": input_channels,
            "out_channels": output_channels,
            "modes": tuple(16 for _ in range(int(dimension))),
            "width": 64,
            "n_layers": 4,
        }
        mode = "field"
    elif model_id == "pibert":
        kwargs = {
            "input_dim": input_channels,
            "output_dim": output_channels,
            "coordinate_dim": int(dimension),
            "hidden_dim": 128,
            "num_layers": 4,
            "num_heads": 8,
        }
        mode = "field_with_coordinates"
    elif model_id == "pinn":
        kwargs = {
            "input_dim": int(dimension),
            "output_dim": output_channels,
            "hidden_channels": 128,
            "depth": 5,
        }
        mode = "coordinates"
        notes.append("PDE, boundary, and initial-condition residuals must be supplied by the experiment")
    elif model_id == "deeponet":
        branch_input_dim = 1
        for size in input_shape:
            branch_input_dim *= size
        kwargs = {
            "branch_input_dim": branch_input_dim,
            "trunk_input_dim": int(dimension),
            "output_dim": output_channels,
            "latent_dim": 128,
        }
        mode = "branch_trunk"
        notes.append("Branch inputs are flattened; use an explicit sensor encoder for very large fields")
    else:
        kwargs = {
            "input_dim": input_channels,
            "output_dim": output_channels,
            "dimension": int(dimension),
        }
        mode = "field"
        notes.append("Generic translation; verify the upstream adapter constructor")

    kwargs.update(dict(overrides or {}))
    return ModelBuildPlan(model_id=model_id, builder_kwargs=kwargs, input_mode=mode, notes=tuple(notes))


__all__ = ["ModelBuildPlan", "translate_model_config"]
