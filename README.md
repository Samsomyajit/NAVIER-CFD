# NAVIER-CFD

**Neural and Agentic Verification, Integration, Evaluation, and Recommendation for Computational Fluid Dynamics**

[![Version](https://img.shields.io/badge/version-0.1.0-2f6f9f.svg)](https://github.com/Samsomyajit/NAVIER-CFD/releases)
[![PyPI](https://img.shields.io/pypi/v/navier-cfd.svg?label=PyPI)](https://pypi.org/project/navier-cfd/)
[![Python](https://img.shields.io/pypi/pyversions/navier-cfd.svg)](https://pypi.org/project/navier-cfd/)
[![License](https://img.shields.io/badge/License-Apache%202.0-4c8c6b.svg)](LICENSE)
[![CI](https://github.com/Samsomyajit/NAVIER-CFD/actions/workflows/ci.yml/badge.svg)](https://github.com/Samsomyajit/NAVIER-CFD/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-6b7280.svg)](https://samsomyajit.github.io/NAVIER-CFD/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-f2c94c.svg)](https://huggingface.co/datasets)
[![Models](https://img.shields.io/badge/models-55-7c6aa6.svg)](#registered-model-families)
[![Datasets](https://img.shields.io/badge/datasets-11-5d8f72.svg)](#why-navier-cfd)

NAVIER-CFD is a CFD-first, uniform Python platform for neural PDE solvers, hybrid numerical acceleration, benchmark datasets, task-aware model recommendation, and agentic experiment planning.

## Scientific pipeline

<p align="center">
  <img src="docs/assets/navier_pipeline.svg" alt="NAVIER-CFD scientific pipeline from CFD task definition and Hugging Face data acquisition to model recommendation, execution, numerical verification and agentic feedback" width="100%">
</p>

The workflow keeps the numerical role of learning explicit—surrogate, closure, corrector, preconditioner, inverse model, controller, or generator—and connects every experiment to versioned data, a traceable model card, CFD-aware metrics, and a reproducible run manifest.

## Why NAVIER-CFD

Neural-solver repositories commonly provide selected architectures and benchmark scripts. NAVIER-CFD adds the workflow and numerical-accountability layer:

- **55-model taxonomy:** acceleration frameworks, surrogates, general PDE solvers, specialized CFD, geometry/unstructured models, foundation models, inverse methods, uncertainty, particle/multiphase, and generative methods.
- **11 first-class datasets:** PDEBench, CFDBench, RealPDEBench, AirfRANS, DrivAerNet++, DrivAerML, The Well, APEBench, ScalarFlow, ShapeNet-Car, and EAGLE.
- **Full Hugging Face support:** discovery, repository inspection, selective downloads, revision pinning, authentication, streaming, caching, and arbitrary CFD dataset IDs.
- **Explainable model recommendation:** hard compatibility filters and transparent ranking by physics, dimension, mesh, geometry, temporal regime, numerical role, memory, conservation, uncertainty, and transfer requirements.
- **Agentic AI:** deterministic offline planning plus a provider-neutral interface for external LLM agents.
- **CFD-aware benchmarking:** field, spectral, rollout, conservation, OOD, quantity-of-interest, uncertainty, wall-clock, memory, and break-even metrics.
- **Safe integration:** external repositories are metadata-first and are never executed automatically.

## Installation

From PyPI:

```bash
pip install navier-cfd
```

From source:

```bash
git clone https://github.com/Samsomyajit/NAVIER-CFD.git
cd NAVIER-CFD
pip install -e .

# development, testing and documentation
pip install -e ".[dev,docs]"
```

## Quick start

```bash
# Explore the catalogs
navier models list
navier models list --category acceleration
navier datasets list

# Search Hugging Face
navier datasets discover "computational fluid dynamics" --limit 20

# Download only the cylinder subset from CFDBench
navier datasets download cfdbench \
  --local-dir ./data/cfdbench \
  --pattern "cylinder/**"

# Recommend models for an unsteady geometry-varying CFD task
navier recommend \
  --problem cylinder_wake \
  --task surrogate \
  --dimension 2 \
  --mesh structured \
  --temporal autoregressive \
  --geometry varying \
  --conservation \
  --top-k 10

# Create a full agentic experiment plan
navier agent plan \
  "Benchmark RealPDEBench cylinder sim-to-real forecasting with 24 GB VRAM, conservation and uncertainty"
```

## Python API

```python
from navier_cfd import Catalog, TaskSpec, recommend_models
from navier_cfd.datasets import HuggingFaceDatasetManager
from navier_cfd.agents import AgentOrchestrator

catalog = Catalog.load_builtin()
task = TaskSpec(
    problem="3d_vehicle_aerodynamics",
    task_type="surrogate",
    dimension=3,
    mesh_type="point_cloud",
    temporal_mode="steady",
    geometry_mode="varying",
    physics=("aerodynamics",),
    requires_geometry_transfer=True,
    requires_mesh_transfer=True,
    hardware_memory_gb=80,
)

for result in recommend_models(task, catalog.models, top_k=8):
    print(result.model.name, result.score, result.reasons, result.cautions)

hf = HuggingFaceDatasetManager()
print(hf.discover("CFD fluid dynamics", limit=10))

plan = AgentOrchestrator(catalog).plan(
    "Accelerate unsteady 3D CFD on unstructured meshes with conservation and fallback"
)
print(plan.to_dict())
```

## Registered model families

- **Physics-informed:** PINN, NSFnets, PINNsFormer, PINO, PI-MFM, RiemannONet.
- **Operator learning:** DeepONet, MIONet, Fourier-DeepONet, Fourier-MIONet, FNO, F-FNO, U-FNO, U-NO, LSM, MWT, Laplace NO, state-space NO.
- **Geometry and transformer solvers:** Geo-FNO, GINO, GNOT, Transolver, UPT, MeshGraphNets, DoMINO, ReViT.
- **CFD-specialized:** PIBERT, FourierFlow, P3D, AeroTransformer, NeuralDEM, DeepM&Mnet, Energy Transformer.
- **Foundation/generative:** DPOT, Poseidon, PROSE-FD, BCAT, PDEformer-1, Tadpole, PDE-Refiner, FunDiff, Flow Matching for PDEs.
- **Acceleration:** Solver-in-the-Loop, INC, PICT, diffSPH, NeuroSEM, neural-operator preconditioned Newton, geometry-aware neural preconditioning.
- **Uncertainty/time adaptation:** Conformalized-DeepONet and TANTE.

“Included” means represented through a uniform, executable model card. Official external implementations remain external so their licenses and upstream revisions are respected.

## Repository map

```text
src/navier_cfd/
  agents/          deterministic and LLM-ready planning
  benchmarks/      CFD metrics and benchmark-plan generation
  datasets/        Hugging Face discovery/download/streaming
  models/          safe adapter protocol
  catalogs.py      55 models and 11 datasets
  recommender.py   explainable task-to-model ranking
case_studies/      eight detailed experimental study guides
configs/tasks/     reusable task specifications
docs/              MkDocs site, model atlas and dataset cards
```

## Design principles

1. State the learned numerical role: surrogate, closure, corrector, preconditioner, inverse model, controller, or generator.
2. Separate interpolation, parameter OOD, geometry transfer, mesh transfer, solver transfer, and sim-to-real transfer.
3. Pin dataset, code, and checkpoint revisions.
4. Keep recommendations explainable and overridable.
5. Never execute third-party code automatically.
6. Evaluate CFD quantities of interest, conservation, stability, and end-to-end cost—not field error alone.

## Case studies

Eight study guides cover PDEBench Navier–Stokes, CFDBench cavity and cylinder flow, RealPDEBench cylinder and FSI, AirfRANS geometry generalization, three-dimensional automotive aerodynamics, and hybrid neural-numerical acceleration.

## Packaging and PyPI

- Package name: `navier-cfd`
- Import namespace: `navier_cfd`
- Current version: `0.1.0`
- Build backend: Hatchling
- Publication: PyPI Trusted Publishing through `.github/workflows/publish-pypi.yml`

A GitHub Release builds, validates, and publishes the wheel and source distribution using OpenID Connect. The PyPI project must authorize this repository and the `pypi` GitHub environment as a trusted publisher before the first release.

## Documentation and CI

- Documentation: https://samsomyajit.github.io/NAVIER-CFD/
- CI validates Python 3.10–3.12.
- Tagged releases build wheel and source distributions.
- GitHub Pages deployment is defined under `.github/workflows/docs.yml`.
- PyPI publication is defined under `.github/workflows/publish-pypi.yml`.

## License

Licensed under the [Apache License 2.0](LICENSE). Cite NAVIER-CFD and the original model, dataset, upstream implementation, and numerical-solver references used in each experiment.
