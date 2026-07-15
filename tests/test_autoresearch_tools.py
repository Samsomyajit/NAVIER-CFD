from __future__ import annotations

from navier_cfd.autoresearch import ToolRegistry


def test_tool_registry_exposes_read_only_research_tools() -> None:
    registry = ToolRegistry()
    names = {spec.name for spec in registry.specs()}
    assert {
        "list_datasets",
        "list_models",
        "plan_research",
        "recommend_models",
        "list_metric_suites",
        "audit_figure_spec",
    } <= names
    assert all(spec.read_only for spec in registry.specs())


def test_tool_registry_audits_figure_spec() -> None:
    registry = ToolRegistry()
    result = registry.call(
        "audit_figure_spec",
        {
            "spec": {
                "figure_type": "truth_prediction_error",
                "fields": ["velocity"],
                "units": "m/s",
                "mask": "fluid_cells",
                "shared_color_limits": True,
            }
        },
    )
    assert result["valid"] is True
