from __future__ import annotations

import numpy as np

from navier_cfd.diagnostics import analyze_interface_error, rank_worst_cases
from navier_cfd.figures import FigureSpec, audit_figure_spec


def test_figure_audit_rejects_misleading_scales() -> None:
    spec = FigureSpec(
        figure_type="truth_prediction_error",
        fields=("velocity_y",),
        units="m/s",
        shared_color_limits=False,
        mask="fluid_cells",
    )
    report = audit_figure_spec(spec)
    assert report.valid is False
    assert any(issue.code == "unshared_truth_prediction_scale" for issue in report.issues)


def test_figure_audit_warns_about_missing_mask_and_vector_output() -> None:
    spec = FigureSpec(
        figure_type="truth_prediction_error",
        fields=("pressure",),
        units="Pa",
        output_formats=("png",),
    )
    report = audit_figure_spec(spec)
    assert report.valid is True
    codes = {issue.code for issue in report.issues}
    assert "mask_not_declared" in codes
    assert "no_vector_output" in codes


def test_interface_error_and_worst_case_diagnostics() -> None:
    target = np.zeros((2, 8, 8), dtype=float)
    prediction = np.zeros_like(target)
    prediction[0, 3:5, 3:5] = 2.0
    prediction[1] = 0.25
    alpha = np.zeros_like(target)
    alpha[:, :, 4:] = 1.0
    analysis = analyze_interface_error(
        prediction,
        target,
        alpha,
        percentile=75,
        spatial_axes=(1, 2),
    )
    assert analysis["interface_fraction"] > 0
    rows = rank_worst_cases(prediction, target, case_ids=("localized", "global"), top_k=2)
    assert rows[0].rmse >= rows[1].rmse
