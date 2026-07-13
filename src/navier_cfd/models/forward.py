from __future__ import annotations

from typing import Any

from ..datasets.core import CFDBatch


def forward_model(model_id: str, model: Any, batch: CFDBatch) -> Any:
    """Dispatch a canonical CFD batch according to the resolved model input mode."""

    mode = getattr(model, "navier_input_mode", None)
    if mode is None:
        if model_id == "pibert":
            mode = "field_with_coordinates"
        elif model_id in {"pinn", "nsfnets"}:
            mode = "coordinates"
        elif model_id == "deeponet":
            mode = "branch_trunk"
        else:
            mode = "field"

    if mode == "coordinates":
        values = batch.coordinates if batch.coordinates is not None else batch.inputs
        return model(values, coordinates=batch.coordinates) if model_id == "nsfnets" else model(values)

    if mode == "branch_trunk":
        if batch.coordinates is None:
            raise ValueError(f"{model_id} requires coordinates in the canonical batch")
        branch = batch.inputs.reshape(batch.inputs.shape[0], -1)
        trunk = batch.coordinates.reshape(batch.coordinates.shape[0], -1, batch.coordinates.shape[-1])
        return model(branch, trunk)

    if mode == "field_with_coordinates":
        return model(batch.inputs, coordinates=batch.coordinates, mask=batch.mask)

    return model(batch.inputs)


__all__ = ["forward_model"]
