from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ..benchmarks.planning import build_benchmark_plan
from ..catalogs import Catalog
from ..recommender import recommend_models
from ..specs import TaskSpec


class LLMBackend(Protocol):
    def __call__(self, *, system: str, user: str, tools: list[dict[str, Any]]) -> dict[str, Any]: ...


@dataclass
class AgentPlan:
    interpreted_task: TaskSpec
    dataset_id: str
    recommended_models: list[str]
    benchmark_plan: dict[str, Any]
    rationale: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentOrchestrator:
    """Provider-neutral experiment planner.

    The deterministic planner works offline. A caller may inject any LLM backend that accepts
    the documented structured interface; no vendor SDK is required by the core package.
    """

    def __init__(self, catalog: Catalog | None = None, backend: LLMBackend | None = None) -> None:
        self.catalog = catalog or Catalog.load_builtin()
        self.backend = backend

    def plan(self, prompt: str) -> AgentPlan:
        task, dataset_id = self._interpret(prompt)
        recommendations = recommend_models(task, self.catalog.models, top_k=8)
        dataset = self.catalog.dataset(dataset_id)
        benchmark = build_benchmark_plan(task, dataset, recommendations)
        rationale = [
            f"Dataset selected from explicit or inferred benchmark: {dataset.name}.",
            "Models were hard-filtered by dimension and mesh compatibility.",
            "Ranking rewards task role, physics overlap, geometry/temporal support, and safety needs.",
            "The emitted plan separates interpolation, OOD, stability, physics, and cost metrics.",
        ]
        return AgentPlan(
            interpreted_task=task,
            dataset_id=dataset_id,
            recommended_models=[item.model.id for item in recommendations],
            benchmark_plan=benchmark.to_dict(),
            rationale=rationale,
        )

    def _interpret(self, prompt: str) -> tuple[TaskSpec, str]:
        text = prompt.lower()
        dataset_id = "pdebench"
        if "realpdebench" in text or "real pde" in text or "sim-to-real" in text:
            dataset_id = "realpdebench"
        elif "cfdbench" in text or "cavity" in text or "dam" in text:
            dataset_id = "cfdbench"

        dimension = 3 if re.search(r"\b3d\b|three-dimensional", text) else 2
        mesh = "unstructured" if "unstructured" in text else "structured"
        temporal = "autoregressive" if any(k in text for k in ["rollout", "forecast", "unsteady", "sequence"]) else "steady"
        geometry = "varying" if any(k in text for k in ["geometry", "shape", "airfoil", "vehicle", "wing"]) else "fixed"
        task_type = "acceleration" if any(k in text for k in ["accelerat", "precondition", "corrector", "solver-in-the-loop"]) else "surrogate"
        problem = "cylinder_wake" if "cylinder" in text else "general_cfd"
        memory = None
        match = re.search(r"(\d+(?:\.\d+)?)\s*gb", text)
        if match:
            memory = float(match.group(1))

        physics = ["incompressible_navier_stokes"]
        if "compressible" in text or "mach" in text or "shock" in text:
            physics = ["compressible_navier_stokes"]
        if "multiphase" in text or "dam" in text:
            physics.append("multiphase")

        task = TaskSpec(
            problem=problem,
            task_type=task_type,
            dimension=dimension,
            mesh_type=mesh,
            temporal_mode=temporal,
            geometry_mode=geometry,
            physics=tuple(physics),
            requires_conservation="conserv" in text or task_type == "acceleration",
            requires_uncertainty="uncertainty" in text or "safe" in text or "certif" in text,
            requires_geometry_transfer="geometry" in text or "shape" in text,
            requires_mesh_transfer="mesh transfer" in text or "cross-mesh" in text,
            requires_long_rollout=temporal == "autoregressive",
            hardware_memory_gb=memory,
            notes=prompt,
        )
        return task, dataset_id

    @staticmethod
    def tool_schemas() -> list[dict[str, Any]]:
        return [
            {"name": "list_datasets", "description": "List registered CFD/PDE datasets."},
            {"name": "inspect_dataset", "description": "Read one dataset card."},
            {"name": "recommend_models", "description": "Rank compatible neural solvers."},
            {"name": "build_benchmark_plan", "description": "Create splits, metrics and ablations."},
            {"name": "emit_run_manifest", "description": "Serialize a reproducible experiment plan."},
        ]

    def plan_json(self, prompt: str) -> str:
        return json.dumps(self.plan(prompt).to_dict(), indent=2)
