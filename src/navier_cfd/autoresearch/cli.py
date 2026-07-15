from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .contracts import ResearchBudget, ResearchMode, StopPolicy
from .mcp_server import run_stdio_server
from .session import AutoResearchSession
from .tools import ToolRegistry

app = typer.Typer(
    help="NAVIER AutoResearch: Codex-integrated, approval-aware CFD research planning and auditing."
)


@app.command("init")
def init_campaign(
    prompt: str = typer.Argument(..., help="Client CFD or chemical-engineering research problem."),
    workspace: Path = typer.Option(Path("navier-autoresearch"), help="Campaign workspace."),
    name: str = typer.Option("NAVIER AutoResearch campaign"),
    domain: str = typer.Option("general_cfd"),
    mode: ResearchMode = typer.Option(ResearchMode.ASSISTANT),
    max_gpu_hours: float = typer.Option(0.0, min=0.0),
    max_cpu_hours: float = typer.Option(0.0, min=0.0),
    max_storage_gb: float = typer.Option(0.0, min=0.0),
    max_download_gb: float = typer.Option(0.0, min=0.0),
    max_experiments: int = typer.Option(0, min=0),
    max_iterations: int = typer.Option(12, min=1),
) -> None:
    session = AutoResearchSession.create(
        workspace,
        prompt,
        name=name,
        domain=domain,
        mode=mode,
        budget=ResearchBudget(
            max_gpu_hours=max_gpu_hours,
            max_cpu_hours=max_cpu_hours,
            max_storage_gb=max_storage_gb,
            max_download_gb=max_download_gb,
            max_experiments=max_experiments,
        ),
        stop_policy=StopPolicy(max_iterations=max_iterations),
    )
    plan = session.plan()
    typer.echo(json.dumps(plan, indent=2, sort_keys=True))
    typer.echo(f"Workspace: {workspace}")


@app.command("plan")
def plan(prompt: str, output: Optional[Path] = None) -> None:
    result = ToolRegistry().call("plan_research", {"prompt": prompt})
    text = json.dumps(result, indent=2, sort_keys=True)
    if output is None:
        typer.echo(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote {output}")


@app.command("tools")
def tools() -> None:
    typer.echo(json.dumps([spec.to_dict() for spec in ToolRegistry().specs()], indent=2, sort_keys=True))


@app.command("mcp")
def mcp() -> None:
    """Run the local NAVIER-CFD MCP server over STDIO for Codex."""

    run_stdio_server()


if __name__ == "__main__":
    app()
