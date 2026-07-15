---
name: navier-run-autoresearch
description: Run or continue a bounded NAVIER-CFD surrogate research campaign from a ResearchContract. Use when the user asks for autonomous research, autoresearch, iterative baselines, ablations, diagnostics, or adaptive replanning; do not bypass approvals, budgets, denied tools, or stopping rules.
---

# Run NAVIER AutoResearch

1. Load the `ResearchContract`, data audit, current session state, prior actions, findings, and decisions.
2. Confirm the campaign mode, remaining budget, allowed tools, approvals, and stop policy.
3. Use the deterministic NAVIER planner and recommender to construct a baseline and ablation campaign.
4. Propose every non-read action through the session before execution.
5. Request approval where the contract requires it.
6. After each completed run, calculate valid numerical and physical metrics, resource usage, and failure diagnostics.
7. Record observed facts, computed results, interpretations, and hypotheses separately.
8. Replan only when evidence supports a new action; do not run arbitrary hyperparameter searches.
9. Evaluate stopping conditions after every iteration.
10. Finish with one of: goal achieved, additional data required, no valid method found, hypothesis rejected, budget exhausted, or campaign stopped by the user.

A negative result is valid. Never conceal failed runs or physics violations.
