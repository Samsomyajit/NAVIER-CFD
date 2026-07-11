# PIBERT

**Registry ID:** `pibert`  
**Categories:** surrogate, physics-informed, specialized CFD  
**Architecture:** bidirectional transformer with hybrid Fourier-wavelet embeddings, physics-biased attention, masked-physics prediction, and equation-consistency pretraining.

## Suitable tasks
Multiscale CFD surrogates, flow reconstruction, and sim-to-real forecasting.

## Demonstrated settings
1D–3D; structured grids; steady and autoregressive regimes; incompressible flow and fluid–structure interaction.

## Strengths
Global/local spectral representation, physics-biased attention, and self-supervised pretraining.

## Cautions
Full 3D training is memory intensive. Benchmark claims should be reproduced with pinned data and code revisions.

## Usage
```bash
navier models info pibert
navier recommend --problem cylinder_wake --task surrogate --dimension 2 --temporal autoregressive
```

## Reference
Chakraborty, Pan & Chen, *PIBERT: A Physics-Informed Bi-directional Hybrid Spectral Transformer for Multiscale CFD Surrogate Modeling*, 2026. Official implementation: https://github.com/Samsomyajit/pibert
