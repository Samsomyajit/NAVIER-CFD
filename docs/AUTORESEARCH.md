# NAVIER AutoResearch and Codex integration

NAVIER-CFD 1.1.0 adds a scientifically governed agent layer for turning a client CFD or chemical-engineering problem into a structured, reproducible research campaign.

The release does **not** grant an LLM unrestricted solver or cluster access. Codex plans, explains, writes code, and calls explicit tools. NAVIER-CFD remains responsible for deterministic dataset contracts, model compatibility, evidence scoring, metric validity, diagnostics, figure specifications, provenance, budgets, and approval boundaries.

## Install

```bash
pip install "navier-cfd[autoresearch]"
```

For a repository checkout:

```bash
pip install -e ".[autoresearch,dev]"
```

## Start a research campaign

```bash
navier-autoresearch init \
  "Reconstruct gas and solids velocities from EP_G history across unseen gas velocities" \
  --workspace runs/bubblenet-autoresearch \
  --domain gas_solid_multiphase \
  --mode guided \
  --max-gpu-hours 24 \
  --max-experiments 12
```

The workspace contains:

```text
research_contract.json
research_plan.json
session_state.json
actions.jsonl
proposals.jsonl
approvals.jsonl
findings.jsonl
decisions.jsonl
```

The contract records the objective, resource budget, allowed and denied tools, approval policy, and stopping rules. The plan reuses the existing deterministic `AgentOrchestrator`, evidence-aware recommender, and benchmark planner.

## Research modes

| Mode | Behavior |
|---|---|
| `assistant` | Every non-read action requires approval. |
| `guided` | Read-only work is automatic; compute, external access, and writes require approval. |
| `bounded` | The agent may act within an explicitly approved tool set, budget, and stopping policy. |

No mode bypasses denied tools or destructive-operation approval.

## Codex MCP integration

Codex supports local STDIO MCP servers from project `.codex/config.toml`. NAVIER-CFD includes a safe read-only MCP server for planning and audit tools.

Copy the example configuration:

```bash
cp .codex/config.toml.example .codex/config.toml
```

Then launch Codex from the trusted repository. The server exposes:

- `list_datasets`
- `list_models`
- `plan_research`
- `recommend_models`
- `list_metric_suites`
- `audit_figure_spec`

The v1.1.0 MCP surface is intentionally read-only. Training, solver execution, cluster submission, and large data transfers will be added only through separately approved tools.

## Repository skills

Codex discovers repository skills under `.agents/skills`. NAVIER-CFD includes skills for:

- defining a research problem;
- auditing CFD data;
- running bounded AutoResearch;
- diagnosing CFD and surrogate results;
- generating research-grade figures;
- reviewing scientific validity.

Skills define **how to conduct a workflow**. MCP tools perform **deterministic operations**.

## Programmatic API

```python
from navier_cfd import AutoResearchSession, ResearchBudget, ResearchMode

session = AutoResearchSession.create(
    "runs/client-project",
    "Predict pressure drop and temperature fields for unseen heat-exchanger geometries",
    domain="heat_transfer",
    mode=ResearchMode.GUIDED,
    budget=ResearchBudget(max_gpu_hours=20, max_experiments=10),
)

plan = session.plan()
print(plan["planner"]["recommended_models"])
```

Propose an action and enforce approval:

```python
from navier_cfd import ActionRisk

proposal = session.propose_action(
    name="Train FNO baseline",
    tool="navier.train_model",
    risk=ActionRisk.COMPUTE,
    reason="Establish a structured-grid neural-operator baseline",
    estimated_cost={"gpu_hours": 2.0},
)

assert not session.can_execute(proposal.id)
session.approve(proposal.id, approved=True, actor="principal_investigator")
```

## Research-grade figure specifications

```python
from navier_cfd import FigureSpec, audit_figure_spec

spec = FigureSpec(
    figure_type="truth_prediction_error",
    fields=("gas_velocity_y",),
    units="m/s",
    shared_color_limits=True,
    error_definition="absolute",
    mask="fluid_cells",
    output_formats=("pdf", "svg", "png"),
)

report = audit_figure_spec(spec)
assert report.valid
```

The audit checks for misleading color scales, missing units, undeclared masks, normalized fields mislabeled as physical units, visual smoothing, cherry-picked cases, low-resolution raster output, and missing vector output.

Optional Matplotlib rendering supports publication-ready truth/prediction/error and profile figures. Every rendering produces a `FigureManifest` sidecar that records the specification and output files.

## CFD diagnostics

The initial deterministic diagnostics include:

- finite field summaries;
- per-case RMSE ranking;
- conditioned RMSE;
- high-gradient phase-interface masks;
- interface-versus-bulk error analysis;
- fraction of total squared error localized near interfaces.

These functions are especially useful for multiphase hidden-field reconstruction, where global accuracy may hide severe local errors around bubble interfaces.

## Scientific boundaries

- AutoResearch cannot manufacture missing experimental evidence.
- Model recommendations remain conditional on available compatibility metadata and paper evidence.
- A generated plan is not proof that a model will perform well.
- Physics metrics are valid only when required coordinates, spacing, channels, units, masks, and reference quantities are available.
- Full autonomous CFD or laboratory control is outside v1.1.0.
