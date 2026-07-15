<p align="center">
  <img src="assets/navier-cfd-logo.svg" alt="NAVIER-CFD logo" width="760">
</p>

# NAVIER-CFD

**Neural and Agentic Verification, Integration, Evaluation, and Recommendation for Computational Fluid Dynamics**

NAVIER-CFD is a CFD-first Python platform for neural PDE solvers, hybrid acceleration, benchmark data, task-aware model recommendation, and agentic experiment planning.

[Open the project website](../){ .md-button .md-button--primary }
[Launch the interactive recommender](../recommender/){ .md-button }
[View the repository](https://github.com/Samsomyajit/NAVIER-CFD){ .md-button }


## What makes it different

- CFD-first taxonomy spanning acceleration, surrogates, general PDE solvers, geometry models, specialized tasks, foundation models, inverse methods, uncertainty, particle/multiphase, and generative models.
- Uniform access to PDEBench, CFDBench, RealPDEBench, AirfRANS, DrivAer datasets, The Well, APEBench, ScalarFlow, and arbitrary Hugging Face datasets.
- Explainable model ranking by dimension, mesh, geometry, temporal regime, physics, memory, conservation, uncertainty, and numerical role.
- Deterministic and LLM-ready experiment planning.
- CFD-aware field, spectral, conservation, rollout, OOD, quantity-of-interest, and computational-cost metrics.
- Metadata-first external adapters that never execute untrusted repositories automatically.

![navier-cfd](./assets/Navier-CFD.png)

## Install

```bash
pip install navier-cfd
```

## Five-minute tour

```bash
navier models list --category geometry
navier datasets list
navier datasets discover "computational fluid dynamics"
navier datasets download cfdbench --local-dir ./data/cfdbench --pattern "cylinder/**"
navier recommend --problem cylinder_wake --task surrogate --dimension 2 --mesh structured --temporal autoregressive --geometry varying --top-k 8
navier agent plan "Benchmark RealPDEBench cylinder sim-to-real forecasting with 24 GB VRAM"
```

## Recommender status

The recommender is operational as an explainable rule-based engine. It applies hard compatibility filters and then ranks candidates by task role, physics, geometry, mesh, temporal mode, rollout requirements, conservation, uncertainty, hardware memory, and integration maturity. See the [recommender documentation](RECOMMENDER.md).
