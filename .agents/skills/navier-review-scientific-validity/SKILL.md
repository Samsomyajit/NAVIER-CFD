---
name: navier-review-scientific-validity
description: Review a NAVIER-CFD campaign as a skeptical scientific auditor. Use for checking baselines, split leakage, equal budgets, metric validity, uncertainty, representative figures, claims, reproducibility, and domain of validity; do not rewrite results to sound stronger than the evidence.
---

# Review scientific validity

1. Verify the research contract and whether the final claim matches the original objective.
2. Inspect data provenance, exclusions, split policy, leakage checks, and generalization definition.
3. Confirm baseline relevance and equal or transparently reported training budgets.
4. Verify metric definitions, axes, masks, units, normalization space, aggregation, and confidence intervals.
5. Check physical validity, numerical stability, uncertainty calibration, and out-of-domain handling.
6. Audit case and figure selection for cherry-picking or inconsistent scales.
7. Confirm that failed runs and negative findings are represented.
8. Check code version, environment, dataset/checkpoint hashes, seeds, manifests, and reproduction commands.
9. Classify each claim as supported, partially supported, unsupported, or outside scope.
10. Recommend concrete corrective actions ordered by scientific importance.

Do not accept a software feature count as evidence of surrogate accuracy or scientific novelty.
