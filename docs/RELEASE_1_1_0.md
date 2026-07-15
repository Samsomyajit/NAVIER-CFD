# NAVIER-CFD 1.1.0

## AutoResearch and Codex integration

NAVIER-CFD 1.1.0 promotes the existing deterministic agentic planner into a persistent, approval-aware research-campaign foundation.

### Added

- `ResearchObjective`, `ResearchContract`, `ResearchBudget`, `StopPolicy`, and research modes;
- persistent `AutoResearchSession` workspaces with action proposals, approvals, findings, resource usage, and stopping decisions;
- local STDIO MCP server for Codex with read-only planning, recommendation, metric, and figure-audit tools;
- repository `AGENTS.md` with CFD scientific-integrity and autonomy rules;
- six Codex skills under `.agents/skills`;
- `navier-autoresearch` command-line entry point;
- `FigureSpec`, `FigureManifest`, scientific figure audit, and optional Matplotlib rendering;
- deterministic field, worst-case, and phase-interface diagnostics;
- focused tests and documentation.

### Safety boundary

The v1.1.0 MCP tools are read-only. Expensive training, solver execution, cluster submission, large downloads, overwrites, and deletion are not exposed as automatic tools. They require future explicit execution adapters with approval and budget enforcement.

### Compatibility

The existing `AgentOrchestrator`, model hub, dataset providers, recommender, trainer, metrics, checkpoints, and experiment API remain available. AutoResearch composes these systems rather than replacing them.
