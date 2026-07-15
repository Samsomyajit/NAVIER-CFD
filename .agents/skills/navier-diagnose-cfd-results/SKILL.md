---
name: navier-diagnose-cfd-results
description: Diagnose CFD or surrogate predictions using deterministic field, temporal, spectral, conservation, profile, uncertainty, interface, and worst-case analysis. Use when asked why a model failed, where errors occur, or whether predictions are physically credible; do not substitute visual inspection for calculations.
---

# Diagnose CFD results

1. Read the run manifest, dataset mapping, normalization, masks, units, coordinates, and checkpoint provenance.
2. Confirm that target and prediction arrays align in case, time, space, and channel axes.
3. Calculate general field metrics in physical units.
4. Calculate only physics metrics whose assumptions are satisfied.
5. Rank worst cases and inspect whether failures cluster by operating condition, geometry, time, regime, or phase.
6. For multiphase data, compare interface-conditioned and bulk errors using a declared interface definition.
7. Check profiles, temporal drift, energy or spectra, boundaries, and uncertainty-error association when applicable.
8. Distinguish global bias, profile collapse, high-frequency loss, boundary error, interface error, instability, and out-of-domain behavior.
9. Generate evidence-linked findings and propose the smallest next experiment that can test the leading hypothesis.

Never call a result physically valid solely because RMSE is low.
