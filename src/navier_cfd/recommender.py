from __future__ import annotations

from dataclasses import dataclass

from .specs import ModelSpec, TaskSpec, normalize_tokens


@dataclass(frozen=True)
class Recommendation:
    model: ModelSpec
    score: float
    reasons: tuple[str, ...]
    cautions: tuple[str, ...]


def _matches(value: str, supported: tuple[str, ...]) -> bool:
    normalized = normalize_tokens(supported)
    return "any" in normalized or value.lower().replace("-", "_") in normalized


def recommend_models(
    task: TaskSpec,
    models: list[ModelSpec],
    top_k: int = 10,
    include_incompatible: bool = False,
) -> list[Recommendation]:
    recommendations: list[Recommendation] = []
    task_physics = normalize_tokens(task.physics)

    for model in models:
        score = 0.0
        reasons: list[str] = []
        cautions: list[str] = []
        compatible = True

        if task.dimension in model.dimensions:
            score += 14
            reasons.append(f"supports {task.dimension}D")
        else:
            compatible = False
            cautions.append(f"does not list {task.dimension}D support")

        if _matches(task.mesh_type, model.mesh_types):
            score += 14
            reasons.append(f"compatible with {task.mesh_type} meshes")
        else:
            compatible = False
            cautions.append(f"mesh mismatch: {task.mesh_type}")

        if _matches(task.temporal_mode, model.temporal_modes):
            score += 10
            reasons.append(f"supports {task.temporal_mode} operation")
        else:
            cautions.append(f"temporal mode not demonstrated: {task.temporal_mode}")

        if _matches(task.geometry_mode, model.geometry_modes):
            score += 12
            reasons.append(f"supports {task.geometry_mode} geometry")
        elif task.requires_geometry_transfer:
            compatible = False
            cautions.append("geometry transfer is required but unsupported")

        model_tasks = normalize_tokens(model.tasks)
        task_type = task.task_type.lower().replace("-", "_")
        if task_type in model_tasks or "general" in model_tasks:
            score += 20
            reasons.append(f"matches task type {task.task_type}")
        elif task_type in normalize_tokens(model.categories):
            score += 15
            reasons.append(f"category matches {task.task_type}")
        else:
            cautions.append(f"task type {task.task_type} is not a primary use case")

        model_physics = normalize_tokens(model.physics)
        overlap = task_physics & model_physics
        if overlap:
            bonus = min(12, 4 * len(overlap))
            score += bonus
            reasons.append("physics overlap: " + ", ".join(sorted(overlap)))
        elif task_physics and "general_pde" not in model_physics:
            cautions.append("no explicit physics-family match")

        categories = set(model.categories)
        if task.requires_conservation:
            if "acceleration" in categories or "physics_informed" in categories or "conservative" in model.tags:
                score += 8
                reasons.append("has a physics/solver pathway for conservation")
            else:
                cautions.append("conservation is not guaranteed by the model card")

        if task.requires_uncertainty:
            if "uncertainty" in categories or "uncertainty" in model.tags:
                score += 8
                reasons.append("uncertainty support is documented")
            else:
                cautions.append("requires an external UQ wrapper")

        if task.requires_long_rollout:
            if "long_rollout" in model.tags or "acceleration" in categories:
                score += 8
                reasons.append("designed or evaluated for long rollouts")
            else:
                cautions.append("long-horizon stability is not a core feature")

        if task.requires_mesh_transfer:
            if "mesh_transfer" in model.tags or "geometry" in categories:
                score += 8
                reasons.append("mesh/geometry transfer is part of the architecture")
            else:
                compatible = False
                cautions.append("mesh transfer is required but not established")

        if task.hardware_memory_gb is not None and model.min_memory_gb is not None:
            if task.hardware_memory_gb >= model.min_memory_gb:
                score += 4
                reasons.append("fits the declared memory budget")
            else:
                compatible = False
                cautions.append(
                    f"estimated minimum memory {model.min_memory_gb:g} GB exceeds budget"
                )

        if task.preferred_framework and model.framework:
            if task.preferred_framework.lower() == model.framework.lower():
                score += 3
                reasons.append(f"uses preferred framework {model.framework}")

        score += {"native": 3, "adapter": 2, "external": 1, "metadata": 0}.get(model.integration, 0)

        if compatible or include_incompatible:
            recommendations.append(
                Recommendation(model=model, score=score, reasons=tuple(reasons), cautions=tuple(cautions))
            )

    recommendations.sort(key=lambda item: (-item.score, item.model.name.lower()))
    return recommendations[:top_k]
