# NAVIER-CFD AAAI-27 Submission Sprint

## Frozen paper question

**Can contract-governed agents improve task-conditioned model selection and scientific reliability in CFD machine-learning workflows?**

## Working title

**NAVIER-CFD: Contract-Governed Agents for Reliable Model Selection and Verification in Computational Fluid Dynamics**

## Primary claims to test

1. A compatibility- and evidence-aware recommender produces lower-regret CFD model shortlists than random-compatible selection, architecture rules, evidence-only ranking, and generic LLM recommendations.
2. Deterministic research contracts reduce unauthorized actions, approval bypasses, budget and stop-policy violations, and unsupported scientific claims compared with prompt-only governance.

The submission must not be framed as a catalogue of models or as a fully autonomous scientist. The model hub, dataset adapters, trainer, metrics, MCP server, skills, and FigureLab are supporting infrastructure.

## Immediate benchmark scope

### Task-conditioned recommendation

Build 12–18 task specifications spanning:

- structured-grid forecasting;
- fixed-geometry CFD surrogates;
- geometry and mesh transfer;
- three-dimensional field prediction;
- multiphase inverse reconstruction;
- generative turbulence; and
- solver correction or preconditioning.

Compare random, random-compatible, rules-only, evidence-only, generic LLM, retrieval-augmented LLM, expert selection, and full NAVIER-CFD. Report Top-1, Recall@3, NDCG@5, Kendall tau, normalized regret, invalid recommendation rate, coverage, and calibration.

### Prospective CFD validation

Highest-priority cases:

1. CFDBench structured fixed-geometry surrogate;
2. AirfRANS geometry and mesh transfer;
3. MFiX/BubbleNet phase-fraction-to-velocity inverse reconstruction.

Train 4–5 compatible models per case under identical screening budgets and splits. Use at least three seeds for the central comparisons where feasible.

### Governance benchmark

Evaluate at least 80–120 scenarios involving safe reads, writes, expensive compute, external calls, destructive actions, budget exhaustion, stopping conditions, prompt injection, conflicting instructions, cherry-picking, and hypotheses presented as findings.

Compare:

1. unrestricted tool-using agent;
2. prompt-only governance;
3. research-contract record without deterministic enforcement;
4. full NAVIER-CFD enforcement.

Report unsafe allows, false denials, budget overruns, approval bypasses, correct stopping, claim-boundary violations, provenance completeness, and recovery success.

## Non-negotiable validity rules

- No prospective task may use its own held-out benchmark result as recommender evidence.
- Candidate models on a task receive the same data split and screening budget.
- Global data-fit metrics and CFD-specific physical diagnostics are reported separately.
- Design-intent smoke labels are not empirical ground truth.
- The read-only v1.1.0 MCP surface must not be described as full solver autonomy.
- Unfinished results remain explicit placeholders; never insert plausible-looking values.

## Go/no-go gate

Proceed with the full submission only when the project has:

- a measured recommender table with meaningful baseline separation;
- a measured governance table from real agent trials;
- at least two prospective CFD cases;
- one case where physical diagnostics change model selection or claim scope; and
- a complete abstract containing measured values rather than promised experiments.
