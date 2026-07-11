from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


_ALLOWED_CATEGORIES = {
    "acceleration",
    "surrogate",
    "general_pde_solver",
    "specialized",
    "geometry",
    "physics_informed",
    "foundation",
    "generative",
    "inverse",
    "uncertainty",
    "particle_multiphase",
}


@dataclass(frozen=True)
class TaskSpec:
    problem: str
    task_type: str
    dimension: int
    mesh_type: str = "structured"
    temporal_mode: str = "steady"
    geometry_mode: str = "fixed"
    physics: tuple[str, ...] = field(default_factory=tuple)
    fidelity: str = "unknown"
    requires_conservation: bool = False
    requires_uncertainty: bool = False
    requires_geometry_transfer: bool = False
    requires_mesh_transfer: bool = False
    requires_long_rollout: bool = False
    hardware_memory_gb: float | None = None
    preferred_framework: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.dimension not in {1, 2, 3}:
            raise ValueError("dimension must be 1, 2, or 3")
        if not self.problem.strip():
            raise ValueError("problem must be non-empty")
        object.__setattr__(self, "physics", tuple(self.physics))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    categories: tuple[str, ...]
    architecture: str
    tasks: tuple[str, ...]
    physics: tuple[str, ...]
    mesh_types: tuple[str, ...]
    geometry_modes: tuple[str, ...]
    temporal_modes: tuple[str, ...]
    dimensions: tuple[int, ...]
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]
    reference: str
    year: int | None = None
    repository: str | None = None
    integration: str = "metadata"
    min_memory_gb: float | None = None
    framework: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        d = dict(data)
        for key in (
            "categories", "tasks", "physics", "mesh_types", "geometry_modes",
            "temporal_modes", "dimensions", "strengths", "limitations", "tags",
        ):
            d[key] = tuple(d.get(key, ()))
        unknown = set(d["categories"]) - _ALLOWED_CATEGORIES
        if unknown:
            raise ValueError(f"Unknown model categories for {d['id']}: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    description: str
    tasks: tuple[str, ...]
    physics: tuple[str, ...]
    dimensions: tuple[int, ...]
    mesh_types: tuple[str, ...]
    geometry_modes: tuple[str, ...]
    temporal_modes: tuple[str, ...]
    hf_repo_id: str | None = None
    hf_config: str | None = None
    source_url: str | None = None
    license: str | None = None
    size: str | None = None
    scenarios: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetSpec":
        d = dict(data)
        for key in (
            "tasks", "physics", "dimensions", "mesh_types", "geometry_modes",
            "temporal_modes", "scenarios", "notes",
        ):
            d[key] = tuple(d.get(key, ()))
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_tokens(values: Iterable[str]) -> set[str]:
    return {str(v).strip().lower().replace("-", "_").replace(" ", "_") for v in values}
