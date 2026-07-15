from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class ResearchMode(str, Enum):
    """How much freedom an AutoResearch session receives."""

    ASSISTANT = "assistant"
    GUIDED = "guided"
    BOUNDED = "bounded"


class ActionRisk(str, Enum):
    """Risk class used by approval and execution policies."""

    READ = "read"
    WRITE = "write"
    COMPUTE = "compute"
    EXTERNAL = "external"
    DESTRUCTIVE = "destructive"


class CampaignStatus(str, Enum):
    CREATED = "created"
    PLANNED = "planned"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class ResearchBudget:
    """Hard resource ceilings for a bounded research campaign."""

    max_gpu_hours: float = 0.0
    max_cpu_hours: float = 0.0
    max_storage_gb: float = 0.0
    max_download_gb: float = 0.0
    max_experiments: int = 0

    def __post_init__(self) -> None:
        numeric = {
            "max_gpu_hours": self.max_gpu_hours,
            "max_cpu_hours": self.max_cpu_hours,
            "max_storage_gb": self.max_storage_gb,
            "max_download_gb": self.max_download_gb,
            "max_experiments": self.max_experiments,
        }
        invalid = {name: value for name, value in numeric.items() if value < 0}
        if invalid:
            raise ValueError(f"Research budgets must be non-negative: {invalid}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StopPolicy:
    """Deterministic stopping conditions for iterative research."""

    min_improvement_percent: float = 1.0
    patience_iterations: int = 3
    max_iterations: int = 12
    stop_on_budget_exhaustion: bool = True
    stop_on_persistent_physics_failure: bool = True

    def __post_init__(self) -> None:
        if self.min_improvement_percent < 0:
            raise ValueError("min_improvement_percent must be non-negative")
        if self.patience_iterations < 1:
            raise ValueError("patience_iterations must be at least 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchObjective:
    """Machine-readable representation of the client's scientific problem."""

    name: str
    prompt: str
    domain: str = "general_cfd"
    inputs: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    generalization: tuple[str, ...] = ()
    success_metrics: tuple[str, ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Research objective name cannot be empty")
        if not self.prompt.strip():
            raise ValueError("Research objective prompt cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["constraints"] = dict(self.constraints)
        return data


@dataclass(frozen=True)
class Hypothesis:
    """One falsifiable statement connected to a proposed experiment."""

    id: str
    statement: str
    rationale: str
    experiment: str
    status: str = "proposed"
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.statement.strip():
            raise ValueError("Hypothesis id and statement are required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchContract:
    """Objective, budget, permissions, and stopping rules for AutoResearch."""

    objective: ResearchObjective
    budget: ResearchBudget = field(default_factory=ResearchBudget)
    stop_policy: StopPolicy = field(default_factory=StopPolicy)
    mode: ResearchMode = ResearchMode.ASSISTANT
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    approval_required_for: tuple[ActionRisk, ...] = (
        ActionRisk.WRITE,
        ActionRisk.COMPUTE,
        ActionRisk.EXTERNAL,
        ActionRisk.DESTRUCTIVE,
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = set(self.allowed_tools) & set(self.denied_tools)
        if overlap:
            raise ValueError(f"Tools cannot be both allowed and denied: {sorted(overlap)}")

    def requires_approval(self, risk: ActionRisk) -> bool:
        if self.mode == ResearchMode.ASSISTANT and risk != ActionRisk.READ:
            return True
        return risk in self.approval_required_for

    def tool_is_allowed(self, tool_name: str) -> bool:
        if tool_name in self.denied_tools:
            return False
        return not self.allowed_tools or tool_name in self.allowed_tools

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective.to_dict(),
            "budget": self.budget.to_dict(),
            "stop_policy": self.stop_policy.to_dict(),
            "mode": self.mode.value,
            "allowed_tools": list(self.allowed_tools),
            "denied_tools": list(self.denied_tools),
            "approval_required_for": [risk.value for risk in self.approval_required_for],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchContract":
        objective_raw = dict(data["objective"])
        objective = ResearchObjective(
            name=str(objective_raw["name"]),
            prompt=str(objective_raw["prompt"]),
            domain=str(objective_raw.get("domain", "general_cfd")),
            inputs=tuple(objective_raw.get("inputs", ())),
            targets=tuple(objective_raw.get("targets", ())),
            generalization=tuple(objective_raw.get("generalization", ())),
            success_metrics=tuple(objective_raw.get("success_metrics", ())),
            constraints=dict(objective_raw.get("constraints", {})),
        )
        budget = ResearchBudget(**dict(data.get("budget", {})))
        stop_policy = StopPolicy(**dict(data.get("stop_policy", {})))
        return cls(
            objective=objective,
            budget=budget,
            stop_policy=stop_policy,
            mode=ResearchMode(str(data.get("mode", ResearchMode.ASSISTANT.value))),
            allowed_tools=tuple(data.get("allowed_tools", ())),
            denied_tools=tuple(data.get("denied_tools", ())),
            approval_required_for=tuple(
                ActionRisk(str(item)) for item in data.get("approval_required_for", ())
            )
            or cls.__dataclass_fields__["approval_required_for"].default,
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ResearchContract":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def objective_from_prompt(
    prompt: str,
    *,
    name: str = "NAVIER AutoResearch campaign",
    domain: str = "general_cfd",
    inputs: Sequence[str] = (),
    targets: Sequence[str] = (),
    generalization: Sequence[str] = (),
    success_metrics: Sequence[str] = (),
    constraints: Mapping[str, Any] | None = None,
) -> ResearchObjective:
    return ResearchObjective(
        name=name,
        prompt=prompt,
        domain=domain,
        inputs=tuple(inputs),
        targets=tuple(targets),
        generalization=tuple(generalization),
        success_metrics=tuple(success_metrics),
        constraints=dict(constraints or {}),
    )
