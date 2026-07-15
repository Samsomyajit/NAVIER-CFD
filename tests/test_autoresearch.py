from __future__ import annotations

from pathlib import Path

import pytest

from navier_cfd.autoresearch import (
    ActionRisk,
    AutoResearchSession,
    ResearchBudget,
    ResearchContract,
    ResearchMode,
    ResearchObjective,
    StopPolicy,
)


class _Plan:
    def to_dict(self):
        return {
            "interpreted_task": {"problem": "fixture"},
            "dataset_id": "pdebench",
            "recommended_models": ["fno"],
            "benchmark_plan": {"splits": ["interpolation", "ood"]},
            "rationale": ["fixture"],
        }


class _Orchestrator:
    def plan(self, prompt: str):
        assert "fluid" in prompt
        return _Plan()


def test_research_contract_roundtrip(tmp_path: Path) -> None:
    contract = ResearchContract(
        objective=ResearchObjective(
            name="Fluid benchmark",
            prompt="Study fluid surrogate transfer",
            inputs=("velocity",),
            targets=("pressure",),
        ),
        budget=ResearchBudget(max_gpu_hours=4, max_experiments=3),
        stop_policy=StopPolicy(max_iterations=5),
        mode=ResearchMode.GUIDED,
        allowed_tools=("navier.plan", "navier.metrics"),
    )
    path = contract.save(tmp_path / "contract.json")
    loaded = ResearchContract.load(path)
    assert loaded == contract
    assert loaded.requires_approval(ActionRisk.COMPUTE)
    assert loaded.tool_is_allowed("navier.plan")
    assert not loaded.tool_is_allowed("navier.download")


def test_session_plans_records_and_enforces_approval(tmp_path: Path) -> None:
    session = AutoResearchSession.create(
        tmp_path,
        "Study fluid surrogate transfer",
        mode=ResearchMode.GUIDED,
        budget=ResearchBudget(max_gpu_hours=2, max_experiments=2),
        orchestrator=_Orchestrator(),
    )
    plan = session.plan()
    assert plan["planner"]["recommended_models"] == ["fno"]
    assert session.plan_path.exists()

    read_action = session.propose_action(
        name="Inspect metrics",
        tool="navier.metrics",
        risk=ActionRisk.READ,
        reason="Establish valid diagnostics",
    )
    assert session.can_execute(read_action.id)

    compute_action = session.propose_action(
        name="Train baseline",
        tool="navier.train",
        risk=ActionRisk.COMPUTE,
        reason="Test the primary hypothesis",
        estimated_cost={"gpu_hours": 1.0},
    )
    assert not session.can_execute(compute_action.id)
    with pytest.raises(PermissionError):
        session.record_action_result(compute_action.id, success=True)
    session.approve(compute_action.id, approved=True, actor="principal_investigator")
    session.record_action_result(compute_action.id, success=True, result={"rmse": 0.1})


def test_session_stops_on_budget_or_patience(tmp_path: Path) -> None:
    session = AutoResearchSession.create(
        tmp_path,
        "Study fluid surrogate transfer",
        budget=ResearchBudget(max_experiments=1),
        stop_policy=StopPolicy(min_improvement_percent=1.0, patience_iterations=2, max_iterations=10),
        orchestrator=_Orchestrator(),
    )
    session.update_usage(experiments=1)
    decision = session.evaluate_iteration(improvement_percent=0.5, physics_valid=True)
    assert decision["should_stop"] is True
    assert any(reason.startswith("budget_exhausted") for reason in decision["reasons"])
