from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from ..datasets import CFDBatch, CFDSample, collate_cfd_samples
from .dataset_factory import load_model
from .forward import forward_model
from .hub import ModelHub


@dataclass(frozen=True)
class AdapterConformanceReport:
    model_id: str
    passed: bool
    runtime_mode: str
    parameter_count: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...] | None
    target_shape: tuple[int, ...]
    checks: Mapping[str, bool]
    dataset_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_model_adapter(
    model_id: str,
    sample: CFDSample,
    *,
    dataset: str | None = None,
    task: Any | None = None,
    hub: ModelHub | None = None,
    overrides: Mapping[str, Any] | None = None,
    forward_fn: Callable[[Any, CFDBatch], Any] | None = None,
    require_backward: bool = True,
) -> AdapterConformanceReport:
    """Construct and exercise one model against a canonical dataset sample.

    Passing means constructor, resolved dataset configuration, forward contract,
    output compatibility, and optional backward propagation worked. It does not
    establish numerical reproduction of the originating paper.
    """

    current_hub = hub or ModelHub()
    status = current_hub.status(model_id)
    checks = {
        "registered": True,
        "executable": status.executable,
        "dependency_available": status.dependency_available,
        "constructed": False,
        "forward": False,
        "shape_compatible": False,
        "backward": False,
    }
    output_shape = None
    parameter_count = 0
    try:
        if not status.executable:
            raise RuntimeError(status.message)
        model, plan = load_model(
            model_id,
            dataset=dataset,
            sample=sample,
            task=task,
            overrides=overrides,
            hub=current_hub,
            return_plan=True,
        )
        checks["constructed"] = True
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        batch = collate_cfd_samples([sample])
        output = (forward_fn or (lambda current, current_batch: forward_model(model_id, current, current_batch)))(
            model,
            batch,
        )
        checks["forward"] = True
        output_shape = tuple(int(value) for value in output.shape)
        target_shape = tuple(int(value) for value in batch.targets.shape)
        checks["shape_compatible"] = output.shape == batch.targets.shape or output.numel() == batch.targets.numel()
        if require_backward:
            output.float().mean().backward()
            checks["backward"] = any(parameter.grad is not None for parameter in model.parameters())
        else:
            checks["backward"] = True
        passed = all(checks.values())
        return AdapterConformanceReport(
            model_id=model_id,
            dataset_id=plan.dataset_id if plan is not None else dataset,
            passed=passed,
            runtime_mode=status.mode,
            parameter_count=parameter_count,
            input_shape=tuple(int(value) for value in batch.inputs.shape),
            output_shape=output_shape,
            target_shape=target_shape,
            checks=checks,
        )
    except Exception as exc:
        target_shape = tuple(int(value) for value in getattr(sample.targets, "shape", ()))
        input_shape = tuple(int(value) for value in getattr(sample.inputs, "shape", ()))
        return AdapterConformanceReport(
            model_id=model_id,
            dataset_id=dataset,
            passed=False,
            runtime_mode=status.mode,
            parameter_count=parameter_count,
            input_shape=input_shape,
            output_shape=output_shape,
            target_shape=target_shape,
            checks=checks,
            error=f"{type(exc).__name__}: {exc}",
        )


__all__ = ["AdapterConformanceReport", "validate_model_adapter"]
