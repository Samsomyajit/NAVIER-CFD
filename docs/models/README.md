# Model atlas

NAVIER-CFD registers 55 reviewed neural-numerical methods in a uniform executable catalog. The catalog covers:

- **Physics-informed:** PINN, NSFnets, PINNsFormer, PINO, PI-MFM, RiemannONet.
- **Deep operator learning:** DeepONet, MIONet, Fourier-DeepONet, Nested Fourier-DeepONet, Fourier-MIONet.
- **Spectral operators:** FNO, F-FNO, U-FNO, U-NO, LSM, MWT, Laplace NO, state-space NO.
- **Geometry and transformers:** Geo-FNO, GINO, GNOT, Galerkin Transformer, FactFormer, ONO, Transolver, UPT, MeshGraphNets, DoMINO, ReViT.
- **CFD-specialized models:** PIBERT, FourierFlow, P3D, AeroTransformer, NeuralDEM, DeepM&Mnet, Energy Transformer.
- **Foundation and generative models:** DPOT, Poseidon, PROSE-FD, BCAT, PDEformer-1, Tadpole, PDE-Refiner, FunDiff, Flow Matching for PDEs.
- **Acceleration frameworks:** Solver-in-the-Loop, INC, PICT, diffSPH, NeuroSEM, neural-operator preconditioned Newton, geometry-aware neural preconditioning.
- **Uncertainty and time adaptation:** Conformalized-DeepONet and TANTE.

Use `navier models list` and `navier models info <id>` for the current machine-readable card. Key architecture cards are included in this directory.
