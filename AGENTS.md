# NAVIER-CFD Codex and AutoResearch Instructions

## Scientific integrity

- Never invent units, field semantics, boundary conditions, solver settings, mesh topology, operating conditions, or experimental metadata.
- Never report a physical metric unless NAVIER-CFD marks it valid and its required metadata are available.
- Evaluate physical metrics after inverse normalization in declared physical units.
- Distinguish observed facts, deterministic calculations, scientific interpretations, and unverified hypotheses.
- Do not silently change official train, validation, or test splits.
- Do not compare models trained with unequal budgets without stating the difference.
- Use shared color limits for truth and prediction fields unless a documented scientific reason requires otherwise.
- Exclude padded, masked, solid, or invalid cells from metrics unless the research contract explicitly includes them.
- Never present an extrapolative or out-of-domain prediction as validated.
- Preserve dataset, source, split, model, checkpoint, metric, figure, and code provenance.

## Agent execution boundaries

- Read-only inspection, deterministic planning, metric validity checks, and figure audits may run automatically.
- Dataset downloads, file writes, training, cluster submission, OpenFOAM, MFiX, DEM, checkpoint replacement, and destructive actions require the approval policy in the active ResearchContract.
- Version 1.1.0 exposes a read-only MCP tool surface by default. Do not bypass it with shell commands or arbitrary code execution.
- Stop when the research contract reaches a resource ceiling, stopping condition, unresolved physics failure, or evidence limitation.

## Repository workflow

- Preserve backward compatibility for the public NAVIER-CFD API unless the task explicitly authorizes a breaking change.
- Add tests for scientific contracts, approval boundaries, diagnostics, plotting, and manifests.
- Keep model recommendations evidence-aware and compatibility-gated.
- Keep optional dependencies lazy so the base package remains usable without Matplotlib or MCP installed.
- Treat generated plots and reports as research artifacts: create traceability sidecars and record selection rules.
