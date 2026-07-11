from navier_cfd.agents import AgentOrchestrator


def test_agent_plan_realpdebench():
    plan = AgentOrchestrator().plan("RealPDEBench cylinder sim-to-real forecast with 24 GB GPU")
    assert plan.dataset_id == "realpdebench"
    assert plan.interpreted_task.problem == "cylinder_wake"
    assert plan.recommended_models
