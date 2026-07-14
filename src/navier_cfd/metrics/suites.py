from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .core import MetricContext, MetricDefinition, MetricResult
from .data import (
    cosine_similarity,
    linfinity,
    mae,
    mse,
    nmse,
    nrmse,
    pearson_r,
    r2,
    relative_l1,
    relative_l2,
    rmse,
    update_ratio,
    variance_scaled_mse,
    variance_scaled_rmse,
)
from .physics import (
    divergence_error,
    kinetic_energy_relative_error,
    mean_velocity_profile_error,
    turbulent_kinetic_energy_error,
    vorticity_rmse,
)
from .spectral import binned_spectral_mse, fourier_rmse_bins, frequency_error, spectral_relative_error


def _realpde_relative_l2(prediction: Any, target: Any, context: MetricContext) -> float:
    return relative_l2(prediction, target, context, per_sample=True)


def _update_ratio_from_context(_: Any, __: Any, context: MetricContext) -> float:
    try:
        fine = int(context.metadata["finetuning_updates"])
        scratch = int(context.metadata["scratch_updates"])
    except KeyError as exc:
        raise ValueError("Update Ratio requires finetuning_updates and scratch_updates metadata") from exc
    return update_ratio(fine, scratch)


METRICS: dict[str, MetricDefinition] = {
    "mse": MetricDefinition("mse", mse, definition="Mean squared error"),
    "rmse": MetricDefinition("rmse", rmse, definition="Root mean squared error"),
    "mae": MetricDefinition("mae", mae, definition="Mean absolute error"),
    "linfinity": MetricDefinition("linfinity", linfinity, definition="Maximum absolute error"),
    "relative_l1": MetricDefinition("relative_l1", relative_l1, definition="Relative L1 error"),
    "relative_l2": MetricDefinition("relative_l2", relative_l2, definition="Relative L2 error"),
    "realpde_relative_l2": MetricDefinition(
        "realpde_relative_l2",
        _realpde_relative_l2,
        definition="Mean per-sample relative L2 error",
    ),
    "nmse": MetricDefinition("nmse", nmse, definition="Mean-square normalized MSE"),
    "nrmse": MetricDefinition("nrmse", nrmse, definition="Root mean-square normalized RMSE"),
    "r2": MetricDefinition("r2", r2, direction="higher", best_value=1.0, definition="Coefficient of determination"),
    "pearson_r": MetricDefinition(
        "pearson_r", pearson_r, direction="higher", best_value=1.0, definition="Pearson correlation"
    ),
    "cosine_similarity": MetricDefinition(
        "cosine_similarity",
        cosine_similarity,
        direction="higher",
        best_value=1.0,
        definition="Cosine similarity",
    ),
    "vmse": MetricDefinition(
        "vmse",
        variance_scaled_mse,
        definition="Variance-scaled mean squared error",
        assumptions=("Uses supplied field variance or target-estimated channel variance.",),
    ),
    "vrmse": MetricDefinition(
        "vrmse",
        variance_scaled_rmse,
        definition="Root variance-scaled mean squared error",
        assumptions=("Uses supplied field variance or target-estimated channel variance.",),
    ),
    "spectral_relative_error": MetricDefinition(
        "spectral_relative_error",
        spectral_relative_error,
        category="spectral",
        definition="Relative error between Fourier magnitudes",
    ),
    "binned_spectral_mse": MetricDefinition(
        "binned_spectral_mse",
        binned_spectral_mse,
        category="spectral",
        definition="Parseval-consistent MSE contributions over radial frequency bins",
    ),
    "frmse": MetricDefinition(
        "frmse",
        fourier_rmse_bins,
        category="physics",
        definition="Fourier-space RMSE in low, middle, and high frequency bands",
    ),
    "frequency_error": MetricDefinition(
        "frequency_error",
        frequency_error,
        category="physics",
        definition="Temporal frequency error of spatially summed signals",
        requires=("time_axis",),
    ),
    "divergence_error": MetricDefinition(
        "divergence_error",
        divergence_error,
        category="physics",
        definition="Absolute difference between predicted and target RMS divergence",
        requires=("velocity_channels",),
    ),
    "kinetic_energy_relative_error": MetricDefinition(
        "kinetic_energy_relative_error",
        kinetic_energy_relative_error,
        category="physics",
        definition="Relative L2 error of kinetic-energy fields",
        requires=("velocity_channels",),
    ),
    "turbulent_kinetic_energy_error": MetricDefinition(
        "turbulent_kinetic_energy_error",
        turbulent_kinetic_energy_error,
        category="physics",
        definition="Absolute error in velocity-fluctuation kinetic energy",
        requires=("velocity_channels", "time_axis"),
    ),
    "mean_velocity_profile_error": MetricDefinition(
        "mean_velocity_profile_error",
        mean_velocity_profile_error,
        category="physics",
        definition="Mean absolute error of long-time velocity profiles or probes",
        requires=("velocity_channels", "time_axis"),
    ),
    "vorticity_rmse": MetricDefinition(
        "vorticity_rmse",
        vorticity_rmse,
        category="physics",
        definition="RMSE of two-dimensional vorticity",
        requires=("velocity_channels",),
    ),
    "update_ratio": MetricDefinition(
        "update_ratio",
        _update_ratio_from_context,
        category="efficiency",
        definition="Finetuning updates divided by scratch-training updates at matched best RMSE",
    ),
}


SUITES: dict[str, tuple[str, ...]] = {
    "data_standard": (
        "mse",
        "rmse",
        "mae",
        "linfinity",
        "relative_l1",
        "relative_l2",
        "nmse",
        "nrmse",
        "r2",
        "pearson_r",
        "cosine_similarity",
    ),
    "the_well": (
        "linfinity",
        "mae",
        "mse",
        "rmse",
        "nmse",
        "nrmse",
        "pearson_r",
        "vmse",
        "vrmse",
        "binned_spectral_mse",
    ),
    "realpdebench": (
        "rmse",
        "mae",
        "realpde_relative_l2",
        "r2",
        "frmse",
        "frequency_error",
        "turbulent_kinetic_energy_error",
        "mean_velocity_profile_error",
        "update_ratio",
    ),
    "fluid_standard": (
        "rmse",
        "relative_l2",
        "spectral_relative_error",
        "divergence_error",
        "kinetic_energy_relative_error",
        "vorticity_rmse",
    ),
}


@dataclass(frozen=True)
class MetricSuite:
    names: tuple[str, ...]
    suite_name: str = "custom"

    @classmethod
    def from_name(cls, name: str) -> "MetricSuite":
        try:
            names = SUITES[name]
        except KeyError as exc:
            raise KeyError(f"Unknown metric suite {name!r}; choose one of {sorted(SUITES)}") from exc
        return cls(names=names, suite_name=name)

    @classmethod
    def combine(cls, names: Iterable[str]) -> "MetricSuite":
        combined: list[str] = []
        labels: list[str] = []
        for name in names:
            labels.append(name)
            members = SUITES.get(name, (name,))
            for member in members:
                if member not in combined:
                    combined.append(member)
        return cls(tuple(combined), suite_name="+".join(labels))

    def evaluate(
        self,
        prediction: Any,
        target: Any,
        *,
        context: MetricContext | None = None,
    ) -> dict[str, MetricResult]:
        context = context or MetricContext()
        results: dict[str, MetricResult] = {}
        for name in self.names:
            try:
                definition = METRICS[name]
            except KeyError as exc:
                raise KeyError(f"Unknown metric {name!r}") from exc
            results[name] = definition.evaluate(prediction, target, context)
        return results

    @staticmethod
    def values(results: Mapping[str, MetricResult], *, include_invalid: bool = False) -> dict[str, Any]:
        return {
            name: result.value
            for name, result in results.items()
            if include_invalid or result.valid
        }

    @staticmethod
    def records(results: Mapping[str, MetricResult]) -> dict[str, dict[str, Any]]:
        return {name: result.to_dict() for name, result in results.items()}


__all__ = ["METRICS", "SUITES", "MetricSuite"]
