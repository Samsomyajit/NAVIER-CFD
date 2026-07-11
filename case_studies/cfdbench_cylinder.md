# CFDBench: cylinder wake

## Goal
Evaluate unsteady wake prediction, phase accuracy, and rollout stability under parameter and geometry changes.

## Recommended models
PIBERT, FourierFlow, FNO/F-FNO, PDE-Refiner, Transolver, UPT, INC for hybrid correction, and a numerical baseline.

## Metrics
Field error, drag/lift coefficients when available, Strouhal number, phase drift, autocorrelation, kinetic-energy spectrum, stable horizon, and blow-up rate.

## CLI
```bash
navier recommend --problem cylinder_wake --task surrogate --dimension 2 --mesh structured --temporal autoregressive --geometry varying --top-k 10
```

## References
CFDBench; PIBERT; PDE-Refiner; INC.
