# Interactive recommender

The NAVIER-CFD project website includes an interactive browser tool at:

`/NAVIER-CFD/recommender/`

## What it does

1. Accepts a CFD task specification.
2. Rejects models that fail hard dimension, mesh, required transfer, or memory constraints.
3. Scores compatible candidates by numerical role, temporal behavior, geometry, physics overlap, conservation pathway, uncertainty support, long-rollout evidence, framework, and integration maturity.
4. Recommends benchmark datasets using dimension, representation, temporal mode, geometry, target-case overlap, and Hugging Face availability.
5. Exports a reproducible JSON run manifest.

## Validation status

The recommender is functional and deterministic. Python tests cover the library implementation, and Node tests cover the browser implementation.

```bash
pytest tests/test_recommender.py
node --test website/recommender/recommender-core.test.mjs
```

The current catalog is architecture-level and partly metadata-level. Therefore, scores indicate **fit for evaluation**, not guaranteed benchmark superiority. A publication-grade recommendation study should add:

- normalized benchmark results;
- uncertainty on rankings;
- dataset-shift descriptors;
- computational cost and break-even estimates;
- calibration against expert model choices;
- prospective validation on unseen CFD tasks.

## Privacy

The browser tool executes locally. Task specifications are not uploaded or stored.
