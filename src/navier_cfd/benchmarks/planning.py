from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..recommender import Recommendation
from ..specs import DatasetSpec, TaskSpec


@dataclass
class BenchmarkPlan:
    task: TaskSpec
    dataset: DatasetSpec
    models: list[str]
    splits: list[str]
    metrics: list[str]
    ablations: list[str]
    reporting: list[str]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_benchmark_plan(
    task: TaskSpec,
    dataset: DatasetSpec,
    recommendations: list[Recommendation],
    max_models: int = 6,
) -> BenchmarkPlan:
    metrics = [
        "relative_l2",
        "rmse",
        "mae",
        "r2",
        "spectral_relative_error",
        "wall_clock",
        "peak_memory",
    ]
    if task.temporal_mode in {"autoregressive", "unsteady", "sequence"}:
        metrics += ["stable_horizon", "rollout_drift", "autocorrelation_error"]
    if task.requires_conservation:
        metrics += ["mass_balance_error", "momentum_balance_error", "divergence_rms"]
    if task.requires_uncertainty:
        metrics += ["coverage", "calibration_error", "risk_coverage"]

    splits = ["interpolation", "parameter_ood"]
    if task.requires_geometry_transfer or task.geometry_mode == "varying":
        splits.append("geometry_ood")
    if task.requires_mesh_transfer:
        splits += ["mesh_resolution_ood", "mesh_generator_ood"]
    if "real" in dataset.name.lower() or "real" in dataset.description.lower():
        splits.append("sim_to_real")

    return BenchmarkPlan(
        task=task,
        dataset=dataset,
        models=[item.model.id for item in recommendations[:max_models]],
        splits=splits,
        metrics=list(dict.fromkeys(metrics)),
        ablations=[
            "remove_physics_features",
            "remove_geometry_features",
            "one_step_vs_unrolled_training",
            "matched_parameter_and_wall_clock_budget",
        ],
        reporting=[
            "three_or_more_random_seeds",
            "confidence_intervals_over_cases",
            "data_generation_cost",
            "training_cost",
            "break_even_query_count",
            "pinned_dataset_and_model_revisions",
        ],
    )
