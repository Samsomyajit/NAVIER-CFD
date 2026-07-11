# PDEBench: 2D Navier–Stokes forecasting

## Goal
Compare general structured-grid PDE solvers under long autoregressive rollout.

## Recommended models
FNO/F-FNO, PIBERT, UPT, DPOT, Poseidon, PDE-Refiner, BCAT, and a U-Net baseline.

## Workflow
```bash
navier datasets info pdebench
navier recommend --problem navier_stokes --task surrogate --dimension 2 --mesh structured --temporal autoregressive --top-k 10
```

## Required metrics
Relative L2, RMSE, spectral error, stable horizon, kinetic-energy spectrum, divergence error, wall-clock, and peak memory.

## Leakage control
Hold out complete PDE parameter regimes and initial-condition families. Do not tune on the final OOD split.

## References
Takamoto et al., *PDEBench: An Extensive Benchmark for Scientific Machine Learning*, NeurIPS Datasets and Benchmarks, 2022.
