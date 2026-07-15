# Agentic AI and NAVIER AutoResearch

NAVIER-CFD separates language-model reasoning from deterministic scientific execution.

The existing provider-neutral `AgentOrchestrator` converts a natural-language CFD objective into a structured task, dataset choice, evidence-aware model shortlist, benchmark plan, and rationale. It can operate offline and does not send data, files, or credentials to an LLM by default.

NAVIER-CFD 1.1.0 adds **NAVIER AutoResearch**, a persistent and approval-aware campaign layer around this planner.

## Current agentic capabilities

1. interpret natural-language CFD objectives;
2. select a relevant registered dataset family;
3. hard-filter model compatibility;
4. rank models using task fit and paper evidence;
5. build benchmark splits, metrics, and ablations;
6. record a research contract, resource budget, approvals, findings, and stopping decisions;
7. expose read-only planning and audit tools to Codex through a local MCP server;
8. provide repository Codex skills for data audit, diagnostics, figures, AutoResearch, and scientific review.

```python
from navier_cfd import AutoResearchSession, ResearchBudget, ResearchMode

session = AutoResearchSession.create(
    "runs/vehicle-aerodynamics",
    "Benchmark unstructured 3D vehicle drag surrogates with geometry holdout",
    domain="external_aerodynamics",
    mode=ResearchMode.GUIDED,
    budget=ResearchBudget(max_gpu_hours=24, max_experiments=8),
)
plan = session.plan()
print(plan["planner"])
```

## Scientific boundary

The language model plans and interprets. NAVIER-CFD calculates, validates, ranks, audits, and records.

The v1.1.0 MCP tools are intentionally read-only. Training, downloads, OpenFOAM/MFiX/DEM execution, Slurm submission, overwrites, and deletion are not automatically exposed. Future execution connectors must use the same ResearchContract approvals, budgets, provenance, and stop policies.

See [NAVIER AutoResearch and Codex integration](AUTORESEARCH.md) for setup and API details.
