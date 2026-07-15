from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from ..agents import AgentOrchestrator
from .contracts import (
    ActionRisk,
    CampaignStatus,
    ResearchBudget,
    ResearchContract,
    ResearchMode,
    ResearchObjective,
    StopPolicy,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


@dataclass
class ResourceUsage:
    gpu_hours: float = 0.0
    cpu_hours: float = 0.0
    storage_gb: float = 0.0
    downloaded_gb: float = 0.0
    experiments: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionProposal:
    id: str
    name: str
    tool: str
    arguments: Mapping[str, Any]
    risk: ActionRisk
    reason: str
    estimated_cost: Mapping[str, float] = field(default_factory=dict)
    requires_approval: bool = True
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["risk"] = self.risk.value
        data["arguments"] = dict(self.arguments)
        data["estimated_cost"] = dict(self.estimated_cost)
        return data


@dataclass(frozen=True)
class Finding:
    kind: str
    statement: str
    evidence: tuple[str, ...] = ()
    confidence: str = "unrated"
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AutoResearchSession:
    """Persistent, approval-aware research campaign state.

    This class deliberately does not execute arbitrary commands. It records the research
    contract, deterministic planner output, action proposals, approvals, resource usage,
    findings, and stopping decisions. Tool execution is delegated to explicit NAVIER-CFD
    integrations or an MCP host such as Codex.
    """

    def __init__(
        self,
        workspace: str | Path,
        contract: ResearchContract,
        *,
        orchestrator: AgentOrchestrator | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.orchestrator = orchestrator or AgentOrchestrator()
        self.status = CampaignStatus.CREATED
        self.iteration = 0
        self.low_improvement_streak = 0
        self.physics_failure_streak = 0
        self.usage = ResourceUsage()
        self._proposals: dict[str, ActionProposal] = {}
        self._approvals: dict[str, bool] = {}
        self._write_contract()
        self._write_state()

    @classmethod
    def create(
        cls,
        workspace: str | Path,
        prompt: str,
        *,
        name: str = "NAVIER AutoResearch campaign",
        domain: str = "general_cfd",
        mode: ResearchMode = ResearchMode.ASSISTANT,
        budget: ResearchBudget | None = None,
        stop_policy: StopPolicy | None = None,
        constraints: Mapping[str, Any] | None = None,
        inputs: tuple[str, ...] = (),
        targets: tuple[str, ...] = (),
        generalization: tuple[str, ...] = (),
        success_metrics: tuple[str, ...] = (),
        allowed_tools: tuple[str, ...] = (),
        denied_tools: tuple[str, ...] = (),
        orchestrator: AgentOrchestrator | None = None,
    ) -> "AutoResearchSession":
        objective = ResearchObjective(
            name=name,
            prompt=prompt,
            domain=domain,
            inputs=inputs,
            targets=targets,
            generalization=generalization,
            success_metrics=success_metrics,
            constraints=dict(constraints or {}),
        )
        contract = ResearchContract(
            objective=objective,
            budget=budget or ResearchBudget(),
            stop_policy=stop_policy or StopPolicy(),
            mode=mode,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
        )
        return cls(workspace, contract, orchestrator=orchestrator)

    @property
    def contract_path(self) -> Path:
        return self.workspace / "research_contract.json"

    @property
    def state_path(self) -> Path:
        return self.workspace / "session_state.json"

    @property
    def plan_path(self) -> Path:
        return self.workspace / "research_plan.json"

    def _write_contract(self) -> None:
        self.contract.save(self.contract_path)

    def _state_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "iteration": self.iteration,
            "low_improvement_streak": self.low_improvement_streak,
            "physics_failure_streak": self.physics_failure_streak,
            "usage": self.usage.to_dict(),
            "workspace": str(self.workspace),
            "updated_at": _utc_now(),
        }

    def _write_state(self) -> None:
        self.state_path.write_text(
            json.dumps(self._state_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    def plan(self) -> dict[str, Any]:
        plan = self.orchestrator.plan(self.contract.objective.prompt).to_dict()
        payload = {
            "research_contract": self.contract.to_dict(),
            "planner": plan,
            "created_at": _utc_now(),
        }
        self.plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.status = CampaignStatus.PLANNED
        self._write_state()
        _append_jsonl(
            self.workspace / "actions.jsonl",
            {
                "event": "research_planned",
                "plan_path": str(self.plan_path),
                "created_at": _utc_now(),
            },
        )
        return payload

    def propose_action(
        self,
        *,
        name: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        risk: ActionRisk = ActionRisk.READ,
        reason: str,
        estimated_cost: Mapping[str, float] | None = None,
    ) -> ActionProposal:
        if not self.contract.tool_is_allowed(tool):
            raise PermissionError(f"Tool {tool!r} is not allowed by the research contract")
        proposal = ActionProposal(
            id=uuid4().hex,
            name=name,
            tool=tool,
            arguments=dict(arguments or {}),
            risk=risk,
            reason=reason,
            estimated_cost=dict(estimated_cost or {}),
            requires_approval=self.contract.requires_approval(risk),
        )
        self._proposals[proposal.id] = proposal
        _append_jsonl(self.workspace / "proposals.jsonl", proposal.to_dict())
        return proposal

    def approve(self, action_id: str, *, approved: bool, actor: str, note: str = "") -> None:
        if action_id not in self._proposals:
            raise KeyError(f"Unknown action proposal: {action_id}")
        self._approvals[action_id] = approved
        _append_jsonl(
            self.workspace / "approvals.jsonl",
            {
                "action_id": action_id,
                "approved": approved,
                "actor": actor,
                "note": note,
                "created_at": _utc_now(),
            },
        )

    def can_execute(self, action_id: str) -> bool:
        proposal = self._proposals[action_id]
        return not proposal.requires_approval or self._approvals.get(action_id, False)

    def record_action_result(
        self,
        action_id: str,
        *,
        success: bool,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if action_id not in self._proposals:
            raise KeyError(f"Unknown action proposal: {action_id}")
        if not self.can_execute(action_id):
            raise PermissionError(f"Action {action_id} has not been approved")
        _append_jsonl(
            self.workspace / "actions.jsonl",
            {
                "event": "action_result",
                "proposal": self._proposals[action_id].to_dict(),
                "success": success,
                "result": dict(result or {}),
                "error": error,
                "created_at": _utc_now(),
            },
        )
        if not success:
            self.status = CampaignStatus.PAUSED
            self._write_state()

    def add_finding(
        self,
        kind: str,
        statement: str,
        *,
        evidence: tuple[str, ...] = (),
        confidence: str = "unrated",
    ) -> Finding:
        finding = Finding(
            kind=kind,
            statement=statement,
            evidence=evidence,
            confidence=confidence,
        )
        _append_jsonl(self.workspace / "findings.jsonl", finding.to_dict())
        return finding

    def update_usage(
        self,
        *,
        gpu_hours: float = 0.0,
        cpu_hours: float = 0.0,
        storage_gb: float = 0.0,
        downloaded_gb: float = 0.0,
        experiments: int = 0,
    ) -> None:
        deltas = (gpu_hours, cpu_hours, storage_gb, downloaded_gb, experiments)
        if any(value < 0 for value in deltas):
            raise ValueError("Resource usage increments must be non-negative")
        self.usage.gpu_hours += gpu_hours
        self.usage.cpu_hours += cpu_hours
        self.usage.storage_gb += storage_gb
        self.usage.downloaded_gb += downloaded_gb
        self.usage.experiments += experiments
        self._write_state()

    def budget_exhausted(self) -> tuple[bool, tuple[str, ...]]:
        budget = self.contract.budget
        checks = {
            "gpu_hours": (self.usage.gpu_hours, budget.max_gpu_hours),
            "cpu_hours": (self.usage.cpu_hours, budget.max_cpu_hours),
            "storage_gb": (self.usage.storage_gb, budget.max_storage_gb),
            "downloaded_gb": (self.usage.downloaded_gb, budget.max_download_gb),
            "experiments": (float(self.usage.experiments), float(budget.max_experiments)),
        }
        exhausted = tuple(
            name for name, (used, limit) in checks.items() if limit > 0 and used >= limit
        )
        return bool(exhausted), exhausted

    def evaluate_iteration(
        self,
        *,
        improvement_percent: float | None,
        physics_valid: bool,
    ) -> dict[str, Any]:
        self.iteration += 1
        policy = self.contract.stop_policy
        if improvement_percent is not None and improvement_percent < policy.min_improvement_percent:
            self.low_improvement_streak += 1
        else:
            self.low_improvement_streak = 0
        if physics_valid:
            self.physics_failure_streak = 0
        else:
            self.physics_failure_streak += 1

        reasons: list[str] = []
        exhausted, resources = self.budget_exhausted()
        if policy.stop_on_budget_exhaustion and exhausted:
            reasons.append("budget_exhausted:" + ",".join(resources))
        if self.iteration >= policy.max_iterations:
            reasons.append("maximum_iterations_reached")
        if self.low_improvement_streak >= policy.patience_iterations:
            reasons.append("insufficient_improvement")
        if (
            policy.stop_on_persistent_physics_failure
            and self.physics_failure_streak >= policy.patience_iterations
        ):
            reasons.append("persistent_physics_failure")

        should_stop = bool(reasons)
        if should_stop:
            self.status = CampaignStatus.STOPPED
        elif self.status in {CampaignStatus.CREATED, CampaignStatus.PLANNED, CampaignStatus.READY}:
            self.status = CampaignStatus.RUNNING
        self._write_state()
        decision = {
            "iteration": self.iteration,
            "improvement_percent": improvement_percent,
            "physics_valid": physics_valid,
            "should_stop": should_stop,
            "reasons": reasons,
            "status": self.status.value,
            "usage": self.usage.to_dict(),
        }
        _append_jsonl(self.workspace / "decisions.jsonl", decision)
        return decision

    def summary(self) -> dict[str, Any]:
        return {
            "contract": self.contract.to_dict(),
            "state": self._state_dict(),
            "plan_exists": self.plan_path.exists(),
            "proposal_count": len(self._proposals),
            "approved_count": sum(self._approvals.values()),
        }
