from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .agents import AgentOrchestrator
from .catalogs import Catalog
from .datasets import HuggingFaceDatasetManager, TheWellDatasetManager
from .evidence import ALGORITHM_VERSION, load_builtin_evidence
from .models import ModelHub, ModelHubError
from .recommender import recommend_models
from .specs import TaskSpec

app = typer.Typer(help="NAVIER-CFD: neural and agentic CFD datasets, models, benchmarks, recommendation, and planning.")
models_app = typer.Typer(help="Browse and load model integrations.")
datasets_app = typer.Typer(help="Discover and download datasets through registered providers.")
evidence_app = typer.Typer(help="Inspect paper-level benchmark evidence used by the recommender.")
agent_app = typer.Typer(help="Agentic experiment planning.")
app.add_typer(models_app, name="models")
app.add_typer(datasets_app, name="datasets")
app.add_typer(evidence_app, name="evidence")
app.add_typer(agent_app, name="agent")
console = Console()


@models_app.command("list")
def models_list(category: Optional[str] = typer.Option(None), query: Optional[str] = None) -> None:
    hub = ModelHub()
    handles = list(hub.models())
    if category:
        handles = [handle for handle in handles if category in handle.spec.categories]
    if query:
        q = query.lower()
        handles = [
            handle
            for handle in handles
            if q in handle.spec.name.lower() or q in " ".join(handle.spec.tags).lower()
        ]
    table = Table(title=f"Models ({len(handles)})")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Categories")
    table.add_column("Runtime")
    table.add_column("Ready")
    for handle in handles:
        status = handle.status
        table.add_row(
            handle.id,
            handle.spec.name,
            ", ".join(handle.spec.categories),
            status.mode,
            "yes" if status.dependency_available else "no",
        )
    console.print(table)


@models_app.command("info")
def models_info(model_id: str) -> None:
    console.print_json(json.dumps(ModelHub().model(model_id).to_dict()))


@models_app.command("executable")
def models_executable(ready_only: bool = typer.Option(False, help="Show only models whose dependencies are installed.")) -> None:
    hub = ModelHub()
    handles = hub.executable(ready_only=ready_only)
    table = Table(title=f"Executable model integrations ({len(handles)})")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Mode")
    table.add_column("Dependency")
    table.add_column("Install spec")
    for handle in handles:
        status = handle.status
        table.add_row(
            handle.id,
            handle.spec.name,
            status.mode,
            "ready" if status.dependency_available else "missing",
            status.install_spec or "—",
        )
    console.print(table)


@models_app.command("install")
def models_install(
    model_id: str,
    allow_external: bool = typer.Option(
        False,
        "--allow-external",
        help="Explicitly allow pip to execute third-party packaging code.",
    ),
    upgrade: bool = typer.Option(False, help="Upgrade an existing dependency."),
) -> None:
    try:
        ModelHub().install(model_id, allow_external=allow_external, upgrade=upgrade)
    except ModelHubError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(f"Installed executable dependencies for {model_id}")


@datasets_app.command("list")
def datasets_list() -> None:
    datasets = Catalog.load_builtin().datasets
    table = Table(title=f"Datasets ({len(datasets)})")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Scenarios")
    table.add_column("Provider")
    table.add_column("Access")
    for dataset in datasets:
        access = dataset.hf_repo_id or dataset.access_base_path or dataset.source_url or "external"
        table.add_row(
            dataset.id,
            dataset.name,
            ", ".join(dataset.scenarios),
            dataset.provider,
            access,
        )
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


@datasets_app.command("well-list")
def datasets_well_list() -> None:
    """List available The Well provider configurations."""

    names = TheWellDatasetManager().list_datasets()
    table = Table(title=f"The Well configurations ({len(names)})")
    table.add_column("well_dataset_name")
    for name in names:
        table.add_row(name)
    console.print(table)


@datasets_app.command("download")
def datasets_download(
    dataset_id: str,
    local_dir: Path = typer.Option(..., help="Destination base directory."),
    configuration: Optional[str] = typer.Option(
        None,
        "--configuration",
        "--well-dataset-name",
        help="Provider-native configuration; required for The Well.",
    ),
    split: Optional[str] = typer.Option(None, help="Provider split, such as train, valid, or test."),
    revision: Optional[str] = None,
    pattern: list[str] = typer.Option(None, "--pattern", help="Repeatable HF allow pattern."),
    endpoint: Optional[str] = None,
    first_only: bool = typer.Option(False, help="The Well: download only the first data file."),
    parallel: bool = typer.Option(False, help="The Well: use parallel curl downloads."),
) -> None:
    dataset = Catalog.load_builtin().dataset(dataset_id)
    if dataset.provider == "the_well":
        if not configuration:
            console.print("[red]The Well requires --configuration <well_dataset_name>.[/red]")
            raise typer.Exit(code=2)
        TheWellDatasetManager().download(
            configuration,
            base_path=local_dir,
            split=split,
            first_only=first_only,
            parallel=parallel,
        )
        console.print_json(
            json.dumps(
                {
                    "provider": "the_well",
                    "dataset_name": configuration,
                    "split": split,
                    "local_path": str(local_dir),
                }
            )
        )
        return
    if split is not None:
        console.print("[yellow]--split is provider-specific and ignored for generic HF snapshots.[/yellow]")
    manager = HuggingFaceDatasetManager(token=os.getenv("HF_TOKEN"), endpoint=endpoint)
    result = manager.download(dataset, local_dir, revision=revision, allow_patterns=pattern or None)
    console.print_json(json.dumps(result.__dict__))


@evidence_app.command("list")
def evidence_list(
    model_id: Optional[str] = typer.Option(None, help="Filter by NAVIER-CFD model ID."),
    metric_group: Optional[str] = typer.Option(None, help="Filter by evidence dimension."),
) -> None:
    records = list(load_builtin_evidence())
    if model_id:
        records = [record for record in records if record.model_id == model_id]
    if metric_group:
        records = [record for record in records if record.metric_group == metric_group]

    table = Table(title=f"Paper evidence ({len(records)}) · algorithm {ALGORITHM_VERSION}")
    table.add_column("Model")
    table.add_column("Benchmark")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Evidence level")
    table.add_column("Paper")
    for record in records:
        table.add_row(
            record.model_id,
            record.benchmark,
            record.metric,
            f"{record.value:g}" + (f" {record.unit}" if record.unit else ""),
            record.evidence_level,
            f"{record.paper_title} ({record.paper_year})",
        )
    console.print(table)


@evidence_app.command("coverage")
def evidence_coverage() -> None:
    records = load_builtin_evidence()
    models = Catalog.load_builtin().models
    counts = Counter(record.model_id for record in records)
    groups = Counter(record.metric_group for record in records)

    table = Table(title=f"Evidence coverage · {len(records)} records")
    table.add_column("Model")
    table.add_column("Registered records", justify="right")
    for model in sorted(models, key=lambda item: (-counts[item.id], item.name.lower())):
        table.add_row(model.name, str(counts[model.id]))
    console.print(table)
    console.print("Metric groups:", dict(sorted(groups.items())))


@app.command("recommend")
def recommend(
    problem: str = typer.Option("general_cfd"),
    task: str = typer.Option("surrogate"),
    dimension: int = typer.Option(2),
    mesh: str = typer.Option("structured"),
    temporal: str = typer.Option("steady"),
    geometry: str = typer.Option("fixed"),
    physics: list[str] = typer.Option(["incompressible_navier_stokes"]),
    fidelity: str = typer.Option("unknown"),
    memory_gb: Optional[float] = typer.Option(None),
    conservation: bool = typer.Option(False),
    uncertainty: bool = typer.Option(False),
    mesh_transfer: bool = typer.Option(False),
    evidence_weight: float = typer.Option(0.70, min=0.0, max=1.0),
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
        fidelity=fidelity,
        hardware_memory_gb=memory_gb,
        requires_conservation=conservation,
        requires_uncertainty=uncertainty,
        requires_mesh_transfer=mesh_transfer,
        requires_geometry_transfer=geometry == "varying",
        requires_long_rollout=temporal in {"autoregressive", "unsteady", "sequence"},
    )
    rows = recommend_models(
        task_spec,
        Catalog.load_builtin().models,
        top_k=top_k,
        evidence_weight=evidence_weight,
    )
    table = Table(title=f"Evidence-aware model recommendations · {ALGORITHM_VERSION}")
    table.add_column("Rank")
    table.add_column("Model")
    table.add_column("Final")
    table.add_column("Evidence")
    table.add_column("Confidence")
    table.add_column("Records")
    table.add_column("Why")
    table.add_column("Cautions")
    for rank, row in enumerate(rows, 1):
        table.add_row(
            str(rank),
            row.model.name,
            f"{row.score:.1f}",
            f"{row.evidence_score:.1f}",
            f"{100 * row.evidence_confidence:.0f}%",
            str(row.evidence_count),
            "; ".join(row.reasons),
            "; ".join(row.cautions),
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
