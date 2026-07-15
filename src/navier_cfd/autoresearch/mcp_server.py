from __future__ import annotations

import json
from typing import Any

from .tools import ToolRegistry


SERVER_INSTRUCTIONS = (
    "NAVIER-CFD tools are scientific decision aids. Use deterministic tools for dataset, model, "
    "metric, and figure claims. Never invent units, boundary conditions, field semantics, or "
    "physics metrics. Current v1.1.0 tools are read-only planning and audit tools; training, solver "
    "execution, downloads, overwrites, and destructive operations require separate approved tools."
)


def build_mcp_server(registry: ToolRegistry | None = None):
    """Build the local STDIO MCP server used by Codex.

    The optional ``mcp`` dependency is intentionally isolated here so importing NAVIER-CFD core
    remains lightweight. Install with ``pip install 'navier-cfd[autoresearch]'``.
    """

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Codex MCP integration requires the optional dependency: "
            "pip install 'navier-cfd[autoresearch]'"
        ) from exc

    tools = registry or ToolRegistry()
    server = FastMCP("NAVIER-CFD AutoResearch", instructions=SERVER_INSTRUCTIONS)

    @server.tool()
    def list_datasets() -> str:
        """List registered CFD/PDE datasets and their task metadata."""

        return json.dumps(tools.call("list_datasets"), indent=2, sort_keys=True)

    @server.tool()
    def list_models() -> str:
        """List registered surrogate model families and compatibility metadata."""

        return json.dumps(tools.call("list_models"), indent=2, sort_keys=True)

    @server.tool()
    def plan_research(prompt: str) -> str:
        """Convert a client CFD problem into a deterministic NAVIER-CFD research plan."""

        return json.dumps(tools.call("plan_research", {"prompt": prompt}), indent=2, sort_keys=True)

    @server.tool()
    def recommend_models(task: dict[str, Any], top_k: int = 8, evidence_weight: float = 0.70) -> str:
        """Rank compatible surrogate models using task fit and paper evidence."""

        return json.dumps(
            tools.call(
                "recommend_models",
                {"task": task, "top_k": top_k, "evidence_weight": evidence_weight},
            ),
            indent=2,
            sort_keys=True,
        )

    @server.tool()
    def list_metric_suites() -> str:
        """List numerical, spectral, and physical metric suites."""

        return json.dumps(tools.call("list_metric_suites"), indent=2, sort_keys=True)

    @server.tool()
    def audit_figure_spec(spec: dict[str, Any]) -> str:
        """Audit a CFD figure specification for scientific and publication integrity."""

        return json.dumps(tools.call("audit_figure_spec", {"spec": spec}), indent=2, sort_keys=True)

    return server


def run_stdio_server() -> None:
    build_mcp_server().run(transport="stdio")
