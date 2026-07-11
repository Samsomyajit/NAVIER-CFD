# Model catalog

The built-in executable registry contains 55 methods and uses multi-label categories.

- **Acceleration:** INC, Solver-in-the-Loop, PICT, diffSPH, NeuroSEM, neural-operator preconditioned Newton, geometry-aware neural preconditioners.
- **Surrogates:** FNO and DeepONet families, PIBERT, FourierFlow, PDE-Refiner, P3D, AeroTransformer, DoMINO.
- **General PDE solvers:** PINNs, DeepONet, FNO/PINO, GNOT, Transolver, UPT, Laplace NO, state-space NO.
- **Geometry:** Geo-FNO, GINO, Transolver, UPT, MeshGraphNets, DoMINO, AeroTransformer, ReViT.
- **Foundation:** DPOT, Poseidon, PROSE-FD, BCAT, PDEformer-1, PI-MFM, P3D, Tadpole.
- **Specialized:** NSFnets, RiemannONet, DeepM&Mnet, NeuralDEM, nested Fourier-DeepONet, Fourier-MIONet.
- **Uncertainty/generative:** Conformalized-DeepONet, FunDiff, flow matching, FourierFlow.

```bash
navier models list
navier models list --category acceleration
navier models info pibert
```

The [model atlas](models/README.md) lists every registered model and provides detailed cards for the most important CFD architectures.
