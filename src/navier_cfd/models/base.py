from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from ..specs import ModelSpec, TaskSpec


@runtime_checkable
class ModelAdapter(Protocol):
    spec: ModelSpec

    def build(self, task: TaskSpec, **kwargs: Any) -> Any: ...
    def validate_task(self, task: TaskSpec) -> list[str]: ...


@dataclass
class ExternalModelAdapter:
    """Metadata-safe adapter for an external implementation.

    It never clones, installs, imports, or executes third-party code automatically.
    """

    spec: ModelSpec

    def build(self, task: TaskSpec, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"{self.spec.name} is an external integration. Follow its official repository "
            "and license, then register an explicit builder with ModelFactory.register()."
        )

    def validate_task(self, task: TaskSpec) -> list[str]:
        problems = []
        if task.dimension not in self.spec.dimensions:
            problems.append(f"{self.spec.name} does not list {task.dimension}D support")
        if task.mesh_type not in self.spec.mesh_types and "any" not in self.spec.mesh_types:
            problems.append(f"{self.spec.name} does not list {task.mesh_type} meshes")
        return problems


class ModelFactory:
    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., Any]] = {}

    def register(self, model_id: str, builder: Callable[..., Any], *, replace: bool = False) -> None:
        if model_id in self._builders and not replace:
            raise ValueError(f"A builder is already registered for {model_id}")
        self._builders[model_id] = builder

    def create(self, spec: ModelSpec, task: TaskSpec, **kwargs: Any) -> Any:
        if spec.id not in self._builders:
            return ExternalModelAdapter(spec).build(task, **kwargs)
        return self._builders[spec.id](task=task, spec=spec, **kwargs)

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))
