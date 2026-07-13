# Model catalog

The built-in executable registry contains 55 methods and uses multi-label categories. The table below cites representative primary papers for the models named in each family. These references identify the originating scientific methods; they do not imply that every NAVIER-CFD native reference implementation is a bit-for-bit reproduction of the authors' code or checkpoints.

| Family | Representative models in NAVIER-CFD | Representative primary citations |
|---|---|---|
| Physics-informed neural fields | PINN, NSFnets, PINNsFormer, PINO, PIBERT | Raissi, Perdikaris & Karniadakis, *JCP* 2019, [Physics-informed neural networks](https://doi.org/10.1016/j.jcp.2018.10.045); Jin et al., *JCP* 2021, [NSFnets](https://doi.org/10.1016/j.jcp.2020.109951); Zhao, Ding & Prakash, [PINNsFormer](https://arxiv.org/abs/2307.11833); Li et al., [PINO](https://arxiv.org/abs/2111.03794); Chakraborty, Pan & Chen, PIBERT (2026), [project source](https://github.com/Samsomyajit/pibert). |
| Branch-trunk and spectral neural operators | DeepONet, MIONet, Fourier-DeepONet, Fourier-MIONet, FNO, F-FNO, U-FNO, U-NO, LSM | Lu et al., *Nature Machine Intelligence* 2021, [DeepONet](https://doi.org/10.1038/s42256-021-00302-5); Jin, Meng & Lu, [MIONet](https://arxiv.org/abs/2202.06137); Li et al., [FNO](https://arxiv.org/abs/2010.08895); Tran et al., [F-FNO](https://arxiv.org/abs/2111.13802); Rahman, Ross & Azizzadenesheli, [U-NO](https://arxiv.org/abs/2204.11127). |
| Geometry, graph and mesh operators | Geo-FNO, GINO, MeshGraphNets, Transolver, GNOT, UPT, DoMINO, ReViT | Li et al., [Geo-FNO](https://arxiv.org/abs/2207.05209); Li et al., [GINO](https://arxiv.org/abs/2309.00583); Pfaff et al., [MeshGraphNets](https://arxiv.org/abs/2010.03409); Wu et al., [Transolver](https://arxiv.org/abs/2402.02366). Cite the corresponding primary paper or official project for GNOT, UPT, DoMINO and ReViT when using those methods. |
| Transformer and foundation-style PDE models | DPOT, Poseidon, PROSE-FD, BCAT, PDEformer-1, PI-MFM, P3D, AeroTransformer, Tadpole | Liu et al., [PROSE-FD](https://arxiv.org/abs/2409.09811); Liu, Sun & Schaeffer, [BCAT](https://arxiv.org/abs/2501.18972); Ye et al., [PDEformer-1](https://arxiv.org/abs/2407.06664). For DPOT, Poseidon, PI-MFM, P3D, AeroTransformer and Tadpole, cite the exact primary paper/version represented by the experiment. |
| Generative and probabilistic PDE models | FourierFlow, PDE-Refiner, FunDiff, Flow Matching for PDEs, Conformalized-DeepONet | Wang et al., [FourierFlow](https://arxiv.org/abs/2506.00862); Lippe et al., [PDE-Refiner](https://arxiv.org/abs/2308.05732). Cite the original FunDiff, flow-matching and conformal-operator papers for experiments using those variants. |
| Hybrid numerical acceleration | Solver-in-the-Loop, INC, PICT, NeuroSEM, neural-operator preconditioned Newton, geometry-aware neural preconditioner | Um et al., [Solver-in-the-Loop](https://arxiv.org/abs/2007.00016); Wei et al., INC (NeurIPS 2025), [official implementation](https://github.com/tum-pbs/INC); Franz et al., PICT, *Journal of Computational Physics* 2025. Cite the exact NeuroSEM and neural-preconditioning papers used in the experiment. |
| Particle and multiphase learning | diffSPH, NeuralDEM, Nested Fourier-DeepONet, Fourier-MIONet, U-FNO | Winchenbach & Thuerey, diffSPH, *Journal of Computational Physics* 2026; Wen et al., U-FNO, *Advances in Water Resources* 2022. Cite the official NeuralDEM and Fourier-operator source associated with the selected implementation and dataset. |
| Specialized coupled and inverse models | DeepM&Mnet, RiemannONet, Energy Transformer flow reconstruction, TANTE | Mao et al., *JCP* 2021, [DeepM&Mnet](https://doi.org/10.1016/j.jcp.2021.110698); Wu et al., TANTE, *JCP* 562 (2026), 115041. Cite the primary RiemannONet and Energy Transformer papers when reporting those results. |

## Citation practice

When publishing results produced with NAVIER-CFD, cite:

1. **NAVIER-CFD** for the integration, dataset configuration, training, evaluation or recommendation workflow;
2. **the original model paper** for each architecture used;
3. **the original dataset or benchmark paper**;
4. **the official upstream implementation and checkpoint**, when applicable.

A model card, native reference implementation, official author implementation and numerical reproduction are different levels of evidence. The `reference` field displayed by the model registry and recommender is intended to preserve this attribution.

```bash
navier models list
navier models list --category acceleration
navier models info pibert
```

The [model atlas](models/README.md) lists every registered model and provides detailed cards for the most important CFD architectures.