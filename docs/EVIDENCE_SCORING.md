# Evidence-aware recommendation

NAVIER-CFD 0.2 replaces a metadata-only model score with a **benchmark-evidence-aware recommendation score**. The objective is not to create a universal leaderboard. It is to estimate how strongly the published evidence for a model transfers to a user-defined CFD task, while exposing uncertainty and missing evidence.

## Why raw paper numbers cannot be pooled directly

An MSE reported on a normalized cylinder-wake dataset is not comparable to an MSE reported on a dimensional automotive-pressure field. Likewise, a drag speedup, a rollout R-squared, and a spectral error describe different numerical goals. NAVIER-CFD therefore stores each result with its benchmark, physical regime, discretization, geometry, temporal setting, fidelity, metric direction, baseline, and provenance.

Citation count, author prestige, venue prestige, and coauthorship centrality are **not performance terms**. Bibliometrics help discover and audit papers; they do not make a model more accurate.

## Evidence record

Each `EvidenceRecord` stores:

- model identifier and paper metadata;
- benchmark and target problem;
- numerical role, physics, dimension, mesh, geometry, temporal regime, and fidelity;
- metric group, raw metric, direction, value, unit, baseline, and relative improvement;
- peer-review status, evidence level, independent evaluation, code/data availability, cases, seeds, and caveats.

The frozen starter catalog is stored in:

```text
src/navier_cfd/data/paper_evidence.json
```

The initial release includes traceable results for PIBERT, FourierFlow, Transolver, GINO, INC, DoMINO, AeroTransformer, P3D, UPT, and FNO. Models without registered evidence remain available, but their evidence score shrinks toward a neutral prior and their confidence is low.

## Stage 1: compatibility gate

Models are first filtered using architecture-level constraints:

- spatial dimension;
- mesh representation;
- geometry and mesh transfer;
- temporal mode;
- numerical role;
- memory budget.

This produces an architecture prior. It is not interpreted as empirical accuracy.

## Stage 2: task-to-evidence similarity

For task `t` and paper result `e`, the transfer similarity is

\[
\rho(t,e)=0.25\rho_P+0.15\rho_d+0.14\rho_m+0.13\rho_g+0.13\rho_\tau+0.13\rho_r+0.07\rho_f,
\]

where the terms represent physics, dimension, mesh, geometry, temporal regime, numerical role, and fidelity. Exact matches receive 1.0; scientifically related settings receive partial credit; unrelated settings receive little weight.

## Metric utility

Same-pipeline baseline comparisons are preferred. For a lower-is-better metric,

\[
g = \frac{b-y}{|b|},
\]

and for a higher-is-better metric,

\[
g = \frac{y-b}{|b|}.
\]

The bounded utility is

\[
u=\frac12+\frac12\tanh(3g).
\]

When no baseline is available, metric-specific transforms are used only when the scale has a clear interpretation, such as R-squared, normalized RMSE, relative L2 error, percent error, or log speedup. Absolute MSE/MAE/RMSE without a same-pipeline baseline remain neutral because their scales depend on units and normalization.

## Evidence quality

Each result receives a quality multiplier based on:

- independent reproduction or official benchmark status;
- peer-reviewed primary paper versus preprint or secondary transcription;
- code and data availability;
- baseline availability;
- reported number of cases and random seeds.

Missing seeds and cases reduce quality. Author-reported results are retained but downweighted rather than silently treated as an official leaderboard.

## Bayesian shrinkage

For metric category `k`, matched results are aggregated with a neutral prior:

\[
\hat{s}_k = \frac{\alpha_0\mu_0 + \sum_i w_i u_i}{\alpha_0+\sum_i w_i},
\qquad w_i=\rho(t,e_i)q(e_i),
\]

where `mu_0 = 0.5` and `alpha_0 = 1.75`. This prevents one isolated paper value from dominating the recommendation.

Evidence confidence is

\[
C=1-\exp\left(-\frac{\sum_i w_i}{2.5}\right).
\]

Coverage is the fraction of task-relevant metric weight for which at least one evidence record is available.

## Task-dependent metric weights

The score changes with numerical role. For example:

- **Surrogate:** field accuracy, QoIs, OOD generalization, physics, rollout, efficiency, scalability, reproducibility.
- **Forecasting:** rollout and spectral accuracy receive more weight.
- **Acceleration/corrector:** matched-error speed, long-horizon stability, and physics consistency dominate.
- **Inverse:** uncertainty, field/QoI accuracy, and OOD robustness receive more weight.
- **Generative:** spectral statistics, distributional uncertainty, and rollout behavior dominate.

Conservation, uncertainty, transfer, three-dimensionality, and long-rollout flags modify these weights.

## Final score

The current default is

\[
S_{final}=100\left(0.30 S_{architecture}+0.70 S_{evidence}\right).
\]

The returned object also contains architecture score, evidence score, confidence, coverage, evidence count, matched paper records, reasons, and cautions. A final score without confidence is incomplete and should not be reported alone.

## CLI

```bash
navier evidence list
navier evidence list --model-id gino
navier evidence coverage

navier recommend \
  --problem vehicle_drag \
  --task surrogate \
  --dimension 3 \
  --mesh point_cloud \
  --geometry varying \
  --physics aerodynamics \
  --fidelity rans \
  --memory-gb 80
```

## Evidence governance

New records should be reviewed as data, not prose. Pull requests must include the primary source, exact table/figure location in `notes`, metric direction, data regime, fidelity, split information, and known comparability limits. Independent reproductions should be separate records rather than overwriting author-reported values.

The next release should add standardized NAVIER-CFD benchmark runs. Those results can be marked `independent_reproduction` and will naturally receive greater evidence weight than unverified cross-paper claims.
