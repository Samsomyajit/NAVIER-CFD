from __future__ import annotations

from typing import Any, Mapping

from ..datasets.core import CFDSample
from ..specs import DatasetSpec, TaskSpec
from .config import ModelBuildPlan, translate_model_config
from .hub import ModelHub, get_model_hub


def _dataset_id(dataset: str | DatasetSpec | None) -> str | None:
    if dataset is None:
        return None
    if isinstance(dataset, DatasetSpec):
        return dataset.id
    return str(dataset)


def configure_model_for_dataset(
    model_id: str,
    dataset: str | DatasetSpec,
    *,
    sample: CFDSample | None = None,
    task: TaskSpec | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ModelBuildPlan:
    """Return the complete reviewable construction plan for a model–dataset pair."""

    return translate_model_config(
        model_id,
        sample,
        dataset_id=_dataset_id(dataset),
        task=task,
        overrides=overrides,
    )


def load_model(
    model_id: str,
    *args: Any,
    dataset: str | DatasetSpec | None = None,
    sample: CFDSample | None = None,
    task: TaskSpec | None = None,
    overrides: Mapping[str, Any] | None = None,
    hub: ModelHub | None = None,
    return_plan: bool = False,
    **kwargs: Any,
) -> Any:
    """Import a model with configuration inferred from a dataset argument.

    Examples
    --------
    ``load_model("fno", dataset="pdebench")`` uses the registered PDEBench
    defaults. Supplying ``sample=canonical_sample`` replaces default channel and
    dimension assumptions with the actual record. Explicit keyword arguments and
    ``overrides`` take final precedence.
    """

    current_hub = hub or get_model_hub()
    dataset_id = _dataset_id(dataset)
    if dataset_id is None and sample is None:
        model = current_hub.load(model_id, *args, task=task, **kwargs)
        return (model, None) if return_plan else model

    plan = translate_model_config(
        model_id,
        sample,
        dataset_id=dataset_id,
        task=task,
        overrides=overrides,
    )
    builder_kwargs = dict(plan.builder_kwargs)
    builder_kwargs.update(kwargs)
    model = current_hub.load(model_id, *args, task=task, **builder_kwargs)

    # PyTorch modules allow metadata attributes; external objects may not.
    for key, value in {
        "navier_model_id": model_id,
        "navier_dataset_id": plan.dataset_id,
        "navier_build_plan": plan,
        "navier_input_mode": plan.input_mode,
        "navier_dataset_configuration": dict(plan.dataset_configuration),
    }.items():
        try:
            setattr(model, key, value)
        except (AttributeError, TypeError):
            pass
    return (model, plan) if return_plan else model


__all__ = ["configure_model_for_dataset", "load_model"]
