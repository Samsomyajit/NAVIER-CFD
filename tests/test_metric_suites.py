from __future__ import annotations

import numpy as np
import pytest

from navier_cfd import MetricContext, MetricSuite, binned_spectral_mse, update_ratio
from navier_cfd.metrics import mse


def _flow() -> np.ndarray:
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    frames = []
    for time in range(4):
        phase = 0.2 * time
        u = np.sin(xx + phase) * np.cos(yy)
        v = -np.cos(xx + phase) * np.sin(yy)
        frames.append(np.stack((u, v), axis=-1))
    return np.asarray(frames, dtype=np.float64)[None, ...]


def test_identity_is_perfect_for_data_and_the_well_suites() -> None:
    target = _flow()
    context = MetricContext(
        sample_axis=0,
        time_axis=1,
        spatial_axes=(2, 3),
        channel_axis=-1,
        velocity_channels=(0, 1),
        spacing=(2.0 * np.pi / 8, 2.0 * np.pi / 8),
        profile_axis=2,
    )
    for suite_name in ("data_standard", "the_well", "fluid_standard"):
        results = MetricSuite.from_name(suite_name).evaluate(target, target, context=context)
        for name, result in results.items():
            assert result.valid, f"{name}: {result.reason}"
            if name in {"r2", "pearson_r", "cosine_similarity"}:
                assert result.value == pytest.approx(1.0)
            elif isinstance(result.value, dict):
                assert all(value == pytest.approx(0.0) for value in result.value.values())
            else:
                assert result.value == pytest.approx(0.0, abs=1e-10)


def test_binned_spectral_mse_is_parseval_consistent() -> None:
    target = np.zeros((1, 16, 16, 1), dtype=np.float64)
    prediction = target.copy()
    prediction[0, :, :, 0] = np.sin(2.0 * np.pi * np.arange(16)[:, None] / 16.0)
    context = MetricContext(sample_axis=0, spatial_axes=(1, 2), channel_axis=-1)
    bands = binned_spectral_mse(prediction, target, context)
    assert bands["total"] == pytest.approx(mse(prediction, target), rel=1e-10, abs=1e-12)
    assert bands["low"] > 0.0


def test_realpdebench_suite_reports_requirements_instead_of_fabricating_metrics() -> None:
    target = _flow()
    context = MetricContext(sample_axis=0, spatial_axes=(2, 3), channel_axis=-1)
    results = MetricSuite.from_name("realpdebench").evaluate(target, target, context=context)
    assert results["rmse"].valid
    assert not results["frequency_error"].valid
    assert not results["turbulent_kinetic_energy_error"].valid
    assert not results["mean_velocity_profile_error"].valid
    assert not results["update_ratio"].valid


def test_realpdebench_physics_metrics_and_update_ratio() -> None:
    target = _flow()
    prediction = 1.1 * target
    context = MetricContext(
        sample_axis=0,
        time_axis=1,
        spatial_axes=(2, 3),
        channel_axis=-1,
        velocity_channels=(0, 1),
        profile_axis=2,
        metadata={"finetuning_updates": 40, "scratch_updates": 100},
    )
    results = MetricSuite.from_name("realpdebench").evaluate(prediction, target, context=context)
    assert all(result.valid for result in results.values())
    assert results["rmse"].value > 0.0
    assert results["frmse"].value["low"] >= 0.0
    assert results["frequency_error"].value > 0.0
    assert results["turbulent_kinetic_energy_error"].value > 0.0
    assert results["mean_velocity_profile_error"].value > 0.0
    assert results["update_ratio"].value == pytest.approx(0.4)
    assert update_ratio(40, 100) == pytest.approx(0.4)
