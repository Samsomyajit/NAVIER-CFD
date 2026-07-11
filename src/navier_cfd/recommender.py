from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .evidence import EvidenceRecord, EvidenceSummary, load_builtin_evidence, score_model_evidence
from .specs import ModelSpec, TaskSpec, normalize_tokens


@dataclass(frozen=True)
class Recommendation:
    model: ModelSpec
    score: float
    architecture_score: float
    evidence_score: float
    evidence_confidence: float
    evidence_coverage: float
    evidence_count: int
    reasons: tuple[str, ...]
    cautions: tuple[str, ...]
    evidence: EvidenceSummary


def _matches(value: str, supported: tuple[str, ...]) -> bool:
    normalized = normalize_tokens(supported)
    return "any" in normalized or value.lower().replace("-", "_") in normalized


def _architecture_score(task: TaskSpec, model: ModelSpec) -> tuple[float, bool, list[str], list[str]]:
    """Compatibility gate and architecture prior.

    This score is deliberately not treated as benchmark performance. It is a
    transparent prior based on whether the architecture can represent the task.
    """

    score = 0.0
    reasons: list[str] = []
    cautions: list[str] = []
    compatible = True
    task_physics = normalize_tokens(task.physics)

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
            cautions.append(f"estimated minimum memory {model.min_memory_gb:g} GB exceeds budget")

    if task.preferred_framework and model.framework:
        if task.preferred_framework.lower() == model.framework.lower():
            score += 3
            reasons.append(f"uses preferred framework {model.framework}")

    score += {"native": 3, "adapter": 2, "external": 1, "metadata": 0}.get(model.integration, 0)
    return score, compatible, reasons, cautions


def recommend_models(
    task: TaskSpec,
    models: list[ModelSpec],
    top_k: int = 10,
    include_incompatible: bool = False,
    evidence_records: Iterable[EvidenceRecord] | None = None,
    evidence_weight: float = 0.70,
) -> list[Recommendation]:
    """Rank models with hard compatibility filters and paper-level evidence.

    The architecture score is a prior. The dominant component is a Bayesian,
    task-similarity-weighted aggregation of traceable paper results. Missing
    evidence shrinks toward a neutral score rather than being interpreted as
    evidence of poor performance.
    """

    records = tuple(evidence_records) if evidence_records is not None else load_builtin_evidence()
    evidence_weight = max(0.0, min(1.0, evidence_weight))
    architecture_weight = 1.0 - evidence_weight
    recommendations: list[Recommendation] = []

    for model in models:
        architecture_score, compatible, reasons, cautions = _architecture_score(task, model)
        evidence = score_model_evidence(task, model.id, records)

        architecture_normalized = min(1.0, max(0.0, architecture_score / 100.0))
        final_score = 100.0 * (
            architecture_weight * architecture_normalized
            + evidence_weight * evidence.score
        )

        if evidence.matched_count:
            reasons.append(
                f"{evidence.matched_count} task-relevant paper results; "
                f"evidence confidence {100 * evidence.confidence:.0f}%"
            )
            for match in evidence.matches[:2]:
                reasons.append(
                    f"evidence: {match.record.benchmark}/{match.record.metric} "
                    f"(similarity {match.task_similarity:.2f}, quality {match.quality:.2f})"
                )
        else:
            cautions.append("no task-relevant quantitative paper evidence is registered yet")

        if evidence.coverage < 0.35:
            cautions.append(
                "paper evidence covers only a small part of the task's required metrics; "
                "treat the rank as provisional"
            )
        if evidence.confidence < 0.40:
            cautions.append("low evidence confidence: independent reproduction or more benchmarks are needed")

        if compatible or include_incompatible:
            recommendations.append(
                Recommendation(
                    model=model,
                    score=final_score,
                    architecture_score=architecture_score,
                    evidence_score=100.0 * evidence.score,
                    evidence_confidence=evidence.confidence,
                    evidence_coverage=evidence.coverage,
                    evidence_count=evidence.matched_count,
                    reasons=tuple(reasons),
                    cautions=tuple(cautions),
                    evidence=evidence,
                )
            )

    recommendations.sort(
        key=lambda item: (
            -item.score,
            -item.evidence_confidence,
            item.model.name.lower(),
        )
    )
    return recommendations[:top_k]
