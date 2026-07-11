from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .agents import AgentOrchestrator
from .catalogs import Catalog
from .datasets import HuggingFaceDatasetManager
from .recommender import recommend_models
from .specs import TaskSpec

app = typer.Typer(help="NAVIER-CFD: neural and agentic CFD datasets, models, benchmarks, recommendation, and planning.")
models_app = typer.Typer(help="Browse model cards.")
datasets_app = typer.Typer(help="Discover and download datasets.")
agent_app = typer.Typer(help="Agentic experiment planning.")
app.add_typer(models_app, name="models")
app.add_typer(datasets_app, name="datasets")
app.add_typer(agent_app, name="agent")
console = Console()


@models_app.command("list")
def models_list(category: Optional[str] = typer.Option(None), query: Optional[str] = None) -> None:
    catalog = Catalog.load_builtin()
    models = catalog.models
    if category:
        models = [m for m in models if category in m.categories]
    if query:
        q = query.lower()
        models = [m for m in models if q in m.name.lower() or q in " ".join(m.tags).lower()]
    table = Table(title=f"Models ({len(models)})")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Categories")
    table.add_column("Integration")
    for m in models:
        table.add_row(m.id, m.name, ", ".join(m.categories), m.integration)
    console.print(table)


@models_app.command("info")
def models_info(model_id: str) -> None:
    model = Catalog.load_builtin().model(model_id)
    console.print_json(json.dumps(model.to_dict()))


@datasets_app.command("list")
def datasets_list() -> None:
    datasets = Catalog.load_builtin().datasets
    table = Table(title=f"Datasets ({len(datasets)})")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Scenarios")
    table.add_column("Hugging Face")
    for d in datasets:
        table.add_row(d.id, d.name, ", ".join(d.scenarios), d.hf_repo_id or "external")
    console.print(table)


@datasets_app.command("info")
def datasets_info(dataset_id: str) -> None:
    dataset = Catalog.load_builtin().dataset(dataset_id)
    console.print_json(json.dumps(dataset.to_dict()))


@datasets_app.command("discover")
def datasets_discover(query: str, limit: int = 20, endpoint: Optional[str] = None) -> None:
    manager = HuggingFaceDatasetManager(token=os.getenv("HF_TOKEN"), endpoint=endpoint)
    table = Table(title=f"Hugging Face dataset search: {query}")
    table.add_column("Repository")
    table.add_column("Downloads")
    table.add_column("Likes")
    for row in manager.discover(query, limit=limit):
        table.add_row(row["id"], str(row["downloads"]), str(row["likes"]))
    console.print(table)


@datasets_app.command("download")
def datasets_download(
    dataset_id: str,
    local_dir: Path = typer.Option(..., help="Destination directory."),
    revision: Optional[str] = None,
    pattern: list[str] = typer.Option(None, "--pattern", help="Repeatable HF allow pattern."),
    endpoint: Optional[str] = None,
) -> None:
    dataset = Catalog.load_builtin().dataset(dataset_id)
    manager = HuggingFaceDatasetManager(token=os.getenv("HF_TOKEN"), endpoint=endpoint)
    result = manager.download(dataset, local_dir, revision=revision, allow_patterns=pattern or None)
    console.print_json(json.dumps(result.__dict__))


@app.command("recommend")
def recommend(
    problem: str = typer.Option("general_cfd"),
    task: str = typer.Option("surrogate"),
    dimension: int = typer.Option(2),
    mesh: str = typer.Option("structured"),
    temporal: str = typer.Option("steady"),
    geometry: str = typer.Option("fixed"),
    physics: list[str] = typer.Option(["incompressible_navier_stokes"]),
    memory_gb: Optional[float] = typer.Option(None),
    conservation: bool = typer.Option(False),
    uncertainty: bool = typer.Option(False),
    mesh_transfer: bool = typer.Option(False),
    top_k: int = typer.Option(10),
) -> None:
    task_spec = TaskSpec(
        problem=problem,
        task_type=task,
        dimension=dimension,
        mesh_type=mesh,
        temporal_mode=temporal,
        geometry_mode=geometry,
        physics=tuple(physics),
        hardware_memory_gb=memory_gb,
        requires_conservation=conservation,
        requires_uncertainty=uncertainty,
        requires_mesh_transfer=mesh_transfer,
        requires_geometry_transfer=geometry == "varying",
        requires_long_rollout=temporal in {"autoregressive", "unsteady", "sequence"},
    )
    rows = recommend_models(task_spec, Catalog.load_builtin().models, top_k=top_k)
    table = Table(title="Task-aware model recommendations")
    table.add_column("Rank")
    table.add_column("Model")
    table.add_column("Score")
    table.add_column("Why")
    table.add_column("Cautions")
    for rank, row in enumerate(rows, 1):
        table.add_row(
            str(rank), row.model.name, f"{row.score:.1f}", "; ".join(row.reasons), "; ".join(row.cautions)
        )
    console.print(table)


@agent_app.command("plan")
def agent_plan(prompt: str, output: Optional[Path] = None) -> None:
    plan = AgentOrchestrator().plan_json(prompt)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(plan, encoding="utf-8")
        console.print(f"Wrote {output}")
    else:
        console.print_json(plan)


@app.command("version")
def version() -> None:
    from . import __version__
    console.print(__version__)


if __name__ == "__main__":
    app()
