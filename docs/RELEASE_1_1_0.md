# NAVIER-CFD 1.1.0

NAVIER-CFD 1.1.0 introduces the first Codex-integrated AutoResearch foundation.

## Main additions

- structured `ResearchObjective`, `ResearchContract`, `ResearchBudget`, and `StopPolicy`;
- persistent `AutoResearchSession` workspaces;
- assistant, guided, and bounded autonomy modes;
- action risk classification and approval recording;
- deterministic budget and stopping decisions;
- a local read-only MCP server;
- six repository Codex skills;
- deterministic field, worst-case, and multiphase-interface diagnostics;
- `FigureSpec`, figure auditing, Matplotlib renderers, and `FigureManifest`;
- repository scientific-integrity rules in `AGENTS.md`;
- project MCP configuration example;
- focused tests and CI integration.

## Documentation

The release includes a complete documentation set:

- [AutoResearch overview](AUTORESEARCH.md)
- [Architecture](AUTORESEARCH_ARCHITECTURE.md)
- [MCP tools](AUTORESEARCH_TOOLS.md)
- [Codex skills](CODEX_SKILLS.md)
- [Contracts and sessions](AUTORESEARCH_SESSIONS.md)
- [CFD diagnostics](CFD_DIAGNOSTICS.md)
- [FigureLab](FIGURELAB.md)

## Validation

The release branch is validated through:

- Python 3.10, 3.11, and 3.12 tests;
- Ruff;
- browser recommender tests;
- native model/provider/metric/training and AutoResearch smoke tests;
- strict documentation build;
- official-source security and live upstream validation.

## Compatibility

Existing dataset providers, model hub APIs, recommendation, training, checkpoints, metrics, experiments, CLI workflows, and the original `AgentOrchestrator` remain available.

AutoResearch composes these systems rather than replacing them.

## Safety and scope

The v1.1.0 MCP tools are intentionally read-only. The release does not claim automatic training, solver execution, cluster submission, active simulation generation, or autonomous laboratory control.

Those capabilities require separate approved execution adapters.
