# Browser recommender

The NAVIER-CFD project website contains a client-side model and dataset recommender.

## Design

- Mirrors the Python compatibility and ranking logic.
- Applies hard filters for dimension, mesh, required transfer, and memory.
- Scores task type, temporal mode, geometry, physics, conservation, uncertainty, rollout, framework, and integration maturity.
- Produces an auditable JSON run manifest.
- Stores and transmits no user data.

## Validation

Run:

```bash
node --test website/recommender/recommender-core.test.mjs
pytest tests/test_recommender.py
```

The website is an explainable rule-based decision-support tool. Its rankings are hypotheses about architectural fit, not evidence of benchmark superiority.
