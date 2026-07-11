from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .specs import TaskSpec, normalize_tokens


ALGORITHM_VERSION = "0.2.0-evidence"
_PRIOR_MEAN = 0.50
_PRIOR_STRENGTH = 1.75


@dataclass(frozen=True)
class EvidenceRecord:
    """One traceable result extracted from one paper or benchmark report.

    Values are never compared blindly across unrelated benchmarks. A record carries
    the task context, metric direction, baseline (when available), provenance and
    reproducibility metadata required to estimate relevance and evidence quality.
    """

    id: str
    model_id: str
    paper_title: str
    paper_year: int
    source_url: str
    benchmark: str
    problem: str
    task_type: str
    metric_group: str
    metric: str
    value: float
    lower_is_better: bool
    physics: tuple[str, ...]
    dimension: int
    mesh_type: str
    geometry_mode: str
    temporal_mode: str
    fidelity: str = "unknown"
    unit: str | None = None
    baseline_value: float | None = None
    baseline_model: str | None = None
    relative_improvement_pct: float | None = None
    evidence_level: str = "preprint_primary"
    peer_reviewed: bool = False
    independent_evaluation: bool = False
    code_available: bool = False
    data_available: bool = False
    n_cases: int | None = None
    n_seeds: int | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceRecord":
        row = dict(data)
        row["physics"] = tuple(row.get("physics", ()))
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceMatch:
    record: EvidenceRecord
    task_similarity: float
    quality: float
    utility: float
    weight: float


@dataclass(frozen=True)
class EvidenceSummary:
    model_id: str
    score: float
    confidence: float
    coverage: float
    effective_evidence: float
    matched_count: int
    category_scores: Mapping[str, float]
    matches: tuple[EvidenceMatch, ...]


_LEVEL_QUALITY = {
    "independent_reproduction": 1.00,
    "official_benchmark": 0.96,
    "peer_reviewed_primary": 0.90,
    "preprint_primary": 0.76,
    "author_reported": 0.62,
    "secondary_summary": 0.48,
}

_TASK_WEIGHTS: dict[str, dict[str, float]] = {
    "surrogate": {
        "field_accuracy": 0.25,
        "qoi_accuracy": 0.13,
        "ood_generalization": 0.16,
        "rollout_stability": 0.10,
        "physics_consistency": 0.10,
        "efficiency": 0.08,
        "scalability": 0.08,
        "reproducibility": 0.10,
    },
    "forecasting": {
        "field_accuracy": 0.20,
        "rollout_stability": 0.23,
        "spectral_accuracy": 0.14,
        "ood_generalization": 0.12,
        "physics_consistency": 0.10,
        "uncertainty": 0.08,
        "efficiency": 0.06,
        "reproducibility": 0.07,
    },
    "acceleration": {
        "efficiency": 0.24,
        "rollout_stability": 0.20,
        "physics_consistency": 0.18,
        "field_accuracy": 0.12,
        "qoi_accuracy": 0.10,
        "ood_generalization": 0.06,
        "scalability": 0.05,
        "reproducibility": 0.05,
    },
    "corrector": {
        "rollout_stability": 0.25,
        "physics_consistency": 0.20,
        "field_accuracy": 0.18,
        "efficiency": 0.15,
        "qoi_accuracy": 0.08,
        "ood_generalization": 0.07,
        "reproducibility": 0.07,
    },
    "preconditioner": {
        "efficiency": 0.32,
        "physics_consistency": 0.22,
        "scalability": 0.14,
        "ood_generalization": 0.10,
        "field_accuracy": 0.08,
        "reproducibility": 0.14,
    },
    "inverse": {
        "field_accuracy": 0.24,
        "qoi_accuracy": 0.16,
        "uncertainty": 0.18,
        "ood_generalization": 0.14,
        "physics_consistency": 0.12,
        "efficiency": 0.06,
        "reproducibility": 0.10,
    },
    "generative": {
        "spectral_accuracy": 0.22,
        "rollout_stability": 0.18,
        "uncertainty": 0.18,
        "field_accuracy": 0.12,
        "physics_consistency": 0.10,
        "ood_generalization": 0.10,
        "efficiency": 0.05,
        "reproducibility": 0.05,
    },
}


def _token(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _expanded_physics(values: Iterable[str]) -> set[str]:
    out = normalize_tokens(values)
    fluid = {
        "fluid_dynamics", "cfd", "navier_stokes", "incompressible_navier_stokes",
        "compressible_navier_stokes", "aerodynamics", "turbulence", "fsi",
        "combustion", "scalar_transport", "free_surface",
    }
    particle = {"particle", "particles", "granular", "multiphase", "dem", "sph"}
    if out & fluid:
        out.update({"fluid_dynamics", "general_pde"})
    if out & particle:
        out.update({"particle", "granular"})
    return out


def _soft_match(a: str, b: str, related: set[frozenset[str]] | None = None) -> float:
    a_n, b_n = _token(a), _token(b)
    if a_n == b_n or "any" in {a_n, b_n}:
        return 1.0
    if related and frozenset({a_n, b_n}) in related:
        return 0.82
    if "unknown" in {a_n, b_n}:
        return 0.65
    return 0.25


def task_similarity(task: TaskSpec, record: EvidenceRecord) -> float:
    task_physics = _expanded_physics(task.physics)
    record_physics = _expanded_physics(record.physics)
    if not task_physics or not record_physics:
        physics = 0.60
    else:
        intersection = len(task_physics & record_physics)
        union = len(task_physics | record_physics)
        physics = max(0.20, intersection / union if union else 0.0)
        if "general_pde" in record_physics and task_physics:
            physics = max(physics, 0.65)

    dimension = 1.0 if task.dimension == record.dimension else 0.15
    mesh = _soft_match(
        task.mesh_type,
        record.mesh_type,
        {
            frozenset({"unstructured", "point_cloud"}),
            frozenset({"meshfree", "point_cloud"}),
        },
    )
    geometry = _soft_match(
        task.geometry_mode,
        record.geometry_mode,
        {frozenset({"varying", "parameterized"})},
    )
    temporal = _soft_match(
        task.temporal_mode,
        record.temporal_mode,
        {
            frozenset({"unsteady", "autoregressive"}),
            frozenset({"sequence", "autoregressive"}),
        },
    )
    role = _soft_match(
        task.task_type,
        record.task_type,
        {
            frozenset({"surrogate", "forecasting"}),
            frozenset({"acceleration", "corrector"}),
            frozenset({"acceleration", "preconditioner"}),
        },
    )
    fidelity = _soft_match(task.fidelity, record.fidelity)

    return (
        0.25 * physics
        + 0.15 * dimension
        + 0.14 * mesh
        + 0.13 * geometry
        + 0.13 * temporal
        + 0.13 * role
        + 0.07 * fidelity
    )


def evidence_quality(record: EvidenceRecord) -> float:
    quality = _LEVEL_QUALITY.get(record.evidence_level, 0.55)
    if record.peer_reviewed:
        quality += 0.04
    if record.independent_evaluation:
        quality += 0.08
    if record.code_available:
        quality += 0.04
    if record.data_available:
        quality += 0.04
    if record.baseline_value is not None or record.relative_improvement_pct is not None:
        quality += 0.03

    if record.n_seeds is None:
        quality *= 0.90
    else:
        quality *= min(1.05, 0.86 + 0.05 * max(1, record.n_seeds))

    if record.n_cases is None:
        quality *= 0.94
    else:
        quality *= 0.78 + 0.22 * (1.0 - math.exp(-max(record.n_cases, 1) / 100.0))

    return max(0.20, min(1.15, quality))


def _relative_gain(record: EvidenceRecord) -> float | None:
    if record.relative_improvement_pct is not None:
        return record.relative_improvement_pct / 100.0
    if record.baseline_value is None or record.baseline_value == 0:
        return None
    if record.lower_is_better:
        return (record.baseline_value - record.value) / abs(record.baseline_value)
    return (record.value - record.baseline_value) / abs(record.baseline_value)


def metric_utility(record: EvidenceRecord) -> float:
    """Map a reported result to [0, 1] without mixing incompatible raw scales."""

    gain = _relative_gain(record)
    if gain is not None:
        return max(0.0, min(1.0, 0.50 + 0.50 * math.tanh(3.0 * gain)))

    metric = _token(record.metric)
    value = float(record.value)

    if metric in {"r2", "r_squared", "drag_r2", "coefficient_of_determination"}:
        return max(0.0, min(1.0, value))
    if metric in {"speedup", "inference_speedup"}:
        return max(0.0, min(1.0, math.log10(max(value, 1.0)) / 4.5))
    if metric in {"percent_error", "relative_percent_error"}:
        return max(0.0, min(1.0, math.exp(-max(value, 0.0) / 5.0)))
    if metric in {"nrmse", "relative_l2", "relative_l_2", "rel_l2"}:
        return max(0.0, min(1.0, math.exp(-max(value, 0.0) / 0.28)))
    if metric in {"max_resolution_linear", "pde_families", "task_families"}:
        # Capability evidence is intentionally capped below a direct accuracy result.
        if metric == "max_resolution_linear":
            return min(0.88, 0.50 + 0.08 * math.log2(max(value, 1.0) / 64.0 + 1.0))
        return min(0.82, 0.45 + 0.06 * math.log2(max(value, 1.0) + 1.0))
    if metric in {"mse", "mae", "rmse"}:
        # Absolute errors are not portable across scaling conventions. Without a
        # same-pipeline baseline they remain neutral rather than creating a false leaderboard.
        return 0.50
    return 0.50


def task_metric_weights(task: TaskSpec) -> dict[str, float]:
    role = _token(task.task_type)
    weights = dict(_TASK_WEIGHTS.get(role, _TASK_WEIGHTS["surrogate"]))

    if task.requires_long_rollout:
        weights["rollout_stability"] = weights.get("rollout_stability", 0.0) + 0.10
    if task.requires_conservation:
        weights["physics_consistency"] = weights.get("physics_consistency", 0.0) + 0.10
    if task.requires_uncertainty:
        weights["uncertainty"] = weights.get("uncertainty", 0.0) + 0.12
    if task.requires_geometry_transfer or task.requires_mesh_transfer:
        weights["ood_generalization"] = weights.get("ood_generalization", 0.0) + 0.10
    if task.dimension == 3:
        weights["scalability"] = weights.get("scalability", 0.0) + 0.06

    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def score_model_evidence(
    task: TaskSpec,
    model_id: str,
    records: Iterable[EvidenceRecord],
    minimum_similarity: float = 0.34,
) -> EvidenceSummary:
    matches: list[EvidenceMatch] = []
    by_category: dict[str, list[EvidenceMatch]] = {}

    for record in records:
        if record.model_id != model_id:
            continue
        similarity = task_similarity(task, record)
        if similarity < minimum_similarity:
            continue
        quality = evidence_quality(record)
        utility = metric_utility(record)
        weight = similarity * quality
        match = EvidenceMatch(record, similarity, quality, utility, weight)
        matches.append(match)
        by_category.setdefault(record.metric_group, []).append(match)

    category_scores: dict[str, float] = {}
    category_effective: dict[str, float] = {}
    for category, category_matches in by_category.items():
        effective = sum(match.weight for match in category_matches)
        posterior = (
            _PRIOR_STRENGTH * _PRIOR_MEAN
            + sum(match.weight * match.utility for match in category_matches)
        ) / (_PRIOR_STRENGTH + effective)
        category_scores[category] = posterior
        category_effective[category] = effective

    weights = task_metric_weights(task)
    evidence_score = 0.0
    covered_weight = 0.0
    for category, category_weight in weights.items():
        evidence_score += category_weight * category_scores.get(category, _PRIOR_MEAN)
        if category in category_scores:
            covered_weight += category_weight

    effective_evidence = sum(match.weight for match in matches)
    confidence = 1.0 - math.exp(-effective_evidence / 2.5)
    coverage = covered_weight

    return EvidenceSummary(
        model_id=model_id,
        score=max(0.0, min(1.0, evidence_score)),
        confidence=max(0.0, min(1.0, confidence)),
        coverage=max(0.0, min(1.0, coverage)),
        effective_evidence=effective_evidence,
        matched_count=len(matches),
        category_scores=category_scores,
        matches=tuple(sorted(matches, key=lambda item: item.weight, reverse=True)),
    )


@lru_cache(maxsize=1)
def load_builtin_evidence() -> tuple[EvidenceRecord, ...]:
    path = Path(__file__).with_name("data") / "paper_evidence.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    records = tuple(EvidenceRecord.from_dict(row) for row in rows)
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evidence record IDs")
    return records
