# AirfRANS: geometry and operating-condition generalization

## Goal
Predict steady RANS fields and integral forces on unseen airfoils, Reynolds numbers, and angles of attack.

## Recommended models
Transolver, GINO, Geo-FNO, UPT, DoMINO, MeshGraphNets, and PointNet/GraphSAGE baselines.

## Metrics
Volume-field error, surface pressure, wall shear, lift, drag, boundary-layer profiles, and OOD degradation.

## Leakage control
Hold out entire geometry families, not random mesh points. Preserve the official extrapolation tasks.

## References
Bonnet et al., *AirfRANS*, NeurIPS Datasets and Benchmarks, 2022.
