from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .specs import FigureSpec


@dataclass(frozen=True)
class FigureAuditIssue:
    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FigureAuditReport:
    valid: bool
    issues: tuple[FigureAuditIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "issues": [issue.to_dict() for issue in self.issues]}


def audit_figure_spec(spec: FigureSpec) -> FigureAuditReport:
    issues: list[FigureAuditIssue] = []
    field_types = {"truth_prediction_error", "profile", "rollout", "spectrum", "interface_error"}
    if spec.figure_type in field_types and not spec.units:
        issues.append(
            FigureAuditIssue(
                "missing_units",
                "error",
                "Physical field figures must declare units or explicitly declare dimensionless values.",
            )
        )
    if spec.figure_type == "truth_prediction_error" and not spec.shared_color_limits:
        issues.append(
            FigureAuditIssue(
                "unshared_truth_prediction_scale",
                "error",
                "Truth and prediction must use the same color limits unless explicitly justified.",
            )
        )
    if spec.metadata.get("normalized", False) and spec.units not in {None, "normalized", "dimensionless"}:
        issues.append(
            FigureAuditIssue(
                "normalized_data_physical_label",
                "error",
                "The figure is marked normalized but carries a physical unit label.",
            )
        )
    if spec.figure_type in {"truth_prediction_error", "interface_error"} and spec.mask is None:
        issues.append(
            FigureAuditIssue(
                "mask_not_declared",
                "warning",
                "Declare how obstacles, padded cells, solids, and invalid regions are masked.",
            )
        )
    if spec.metadata.get("cherry_picked", False):
        issues.append(
            FigureAuditIssue(
                "cherry_picked_cases",
                "error",
                "Selected cases are marked cherry-picked; include a representative selection rule.",
            )
        )
    if spec.metadata.get("interpolation") not in {None, "none", "nearest"}:
        issues.append(
            FigureAuditIssue(
                "visual_smoothing",
                "warning",
                "Interpolation may visually smooth CFD errors; record and justify it.",
            )
        )
    if "png" in spec.output_formats and spec.dpi < 300:
        issues.append(
            FigureAuditIssue(
                "low_raster_resolution",
                "warning",
                "Raster output below 300 dpi is not recommended for publication.",
            )
        )
    if not ({"pdf", "svg"} & set(spec.output_formats)):
        issues.append(
            FigureAuditIssue(
                "no_vector_output",
                "warning",
                "Include PDF or SVG for vector text and line art.",
            )
        )
    valid = not any(issue.severity == "error" for issue in issues)
    return FigureAuditReport(valid=valid, issues=tuple(issues))
