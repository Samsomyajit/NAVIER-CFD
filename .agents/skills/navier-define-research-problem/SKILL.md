---
name: navier-define-research-problem
description: Define a CFD or chemical-engineering surrogate research objective, contract, hypotheses, success criteria, budget, permissions, and stopping rules. Use when a client describes a new problem or asks to start NAVIER AutoResearch; do not use for merely explaining an already completed result.
---

# Define a NAVIER AutoResearch problem

1. Restate the scientific objective without inflating the claim.
2. Identify inputs, targets, geometry, mesh representation, temporal mode, physics, solver or experiment source, operating variables, intended generalization, and compute constraints.
3. Separate confirmed metadata from assumptions and missing information.
4. Create a `ResearchObjective` and `ResearchContract`.
5. Choose the least autonomous suitable mode: `assistant`, `guided`, or `bounded`.
6. Define resource ceilings and actions requiring approval.
7. Define measurable success criteria and deterministic stopping rules.
8. Form falsifiable hypotheses, each linked to a specific experiment or ablation.
9. Save the contract before proposing compute or external data access.

Never promise that the requested surrogate is feasible before the data audit and baselines are complete.
