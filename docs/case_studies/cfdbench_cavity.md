# CFDBench: lid-driven cavity flow

## Goal
Test generalization across boundary conditions and physical properties in a canonical incompressible CFD problem.

## Data
```bash
navier datasets download cfdbench --local-dir ./data/cfdbench --pattern "cavity/**"
```

## Recommended models
FNO, U-NO, Transolver, PIBERT, DeepONet, and U-Net.

## Metrics
Velocity and pressure MSE, relative L2, divergence RMS, recirculation-center location, wall shear, and matched-error latency.

## Study design
Use separate BC, property, and combined OOD splits. Report whether the data are raw or interpolated to 64×64.

## References
Luo, Chen & Zhang, *CFDBench: A Large-Scale Benchmark for Machine Learning Methods in Fluid Dynamics*, 2023.
