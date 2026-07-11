<p align="center">
  <img src="docs/assets/navier-cfd-logo.svg" alt="NAVIER-CFD logo" width="860">
</p>

<h1 align="center">NAVIER-CFD</h1>

<p align="center"><strong>Neural and Agentic Verification, Integration, Evaluation, and Recommendation for Computational Fluid Dynamics</strong></p>

<p align="center">
  <a href="https://github.com/Samsomyajit/NAVIER-CFD/releases"><img src="https://img.shields.io/badge/version-0.1.0-2f6f9f.svg" alt="Version"></a>
  <a href="https://pypi.org/project/navier-cfd/"><img src="https://img.shields.io/pypi/v/navier-cfd.svg?label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/navier-cfd/"><img src="https://img.shields.io/pypi/pyversions/navier-cfd.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-4c8c6b.svg" alt="Apache 2.0"></a>
  <a href="https://github.com/Samsomyajit/NAVIER-CFD/actions/workflows/ci.yml"><img src="https://github.com/Samsomyajit/NAVIER-CFD/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://samsomyajit.github.io/NAVIER-CFD/"><img src="https://img.shields.io/badge/project-website-0d6fdc.svg" alt="Project website"></a>
  <a href="https://samsomyajit.github.io/NAVIER-CFD/recommender/"><img src="https://img.shields.io/badge/tool-recommender-13b7d8.svg" alt="Interactive recommender"></a>
  <img src="https://img.shields.io/badge/models-55-7c6aa6.svg" alt="55 models">
  <img src="https://img.shields.io/badge/datasets-11-5d8f72.svg" alt="11 datasets">
</p>

NAVIER-CFD is a CFD-first Python platform and project website for neural PDE solvers, hybrid numerical acceleration, benchmark datasets, explainable task-aware recommendation, Hugging Face integration, and agentic experiment planning.

## Project website and interactive tool

- **Project website:** https://samsomyajit.github.io/NAVIER-CFD/
- **Interactive model and dataset recommender:** https://samsomyajit.github.io/NAVIER-CFD/recommender/
- **Technical documentation:** https://samsomyajit.github.io/NAVIER-CFD/docs/

The web recommender runs entirely in the browser. It mirrors the Python ranking rules, applies hard compatibility filters, explains every score, recommends suitable benchmark datasets, and exports a reproducible run manifest.

> The recommender is a deterministic architecture-compatibility decision-support system. It is functional and tested, but its ranking should be treated as a hypothesis to validate on the selected CFD benchmark, discretization, and quantities of interest—not as proof that one model is universally superior.

## Scientific pipeline

<p align="center">
  <img src="docs/assets/navier_pipeline.svg" alt="NAVIER-CFD scientific pipeline" width="100%">
</p>

The workflow keeps the learned numerical role explicit—surrogate, closure, corrector, preconditioner, inverse model, controller, or generator—and connects each experiment to versioned data, traceable model cards, CFD-aware metrics, and a reproducible run manifest.

## Why NAVIER-CFD

- **55-model taxonomy:** acceleration frameworks, surrogates, general PDE solvers, specialized CFD, geometry and unstructured-mesh models, foundation models, inverse methods, uncertainty, particle and multiphase models, and generative methods.
- **11 first-class datasets:** PDEBench, CFDBench, RealPDEBench, AirfRANS, DrivAerNet++, DrivAerML, The Well, APEBench, ScalarFlow, ShapeNet-Car, and EAGLE.
- **Full Hugging Face support:** discovery, inspection, selective downloads, revision pinning, authentication, streaming, caching, and arbitrary CFD dataset identifiers.
- **Explainable recommendation:** compatibility filtering and transparent ranking by physics, dimension, mesh, geometry, temporal regime, numerical role, memory, conservation, uncertainty, and transfer requirements.
- **Agentic AI:** deterministic offline planning and provider-neutral interfaces for external LLM agents.
- **CFD-aware benchmarking:** field, spectral, rollout, conservation, OOD, quantity-of-interest, uncertainty, wall-clock, memory, and break-even metrics.
- **Safe integration:** external repositories remain metadata-first and are never executed automatically.

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

# development, tests, and documentation
pip install -e ".[dev,docs]"
```

## Quick start

```bash
# Explore catalogs
navier models list
navier models list --category acceleration
navier datasets list

# Search Hugging Face
navier datasets discover "computational fluid dynamics" --limit 20

# Download a CFDBench subset
navier datasets download cfdbench \
  --local-dir ./data/cfdbench \
  --pattern "cylinder/**"

# Recommend models
navier recommend \
  --problem cylinder_wake \
  --task surrogate \
  --dimension 2 \
  --mesh structured \
  --temporal autoregressive \
  --geometry varying \
  --conservation \
  --top-k 10

# Generate an agentic experiment plan
navier agent plan \
  "Benchmark RealPDEBench cylinder sim-to-real forecasting with 24 GB VRAM, conservation and uncertainty"
```

## Python API

```python
from navier_cfd import Catalog, TaskSpec, recommend_models

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
    print(result.model.name, result.score)
    print("reasons:", result.reasons)
    print("cautions:", result.cautions)
```

## Registered model families

- **Physics-informed:** PINN, NSFnets, PINNsFormer, PINO, PI-MFM, and RiemannONet.
- **Operator learning:** DeepONet, MIONet, Fourier-DeepONet, Fourier-MIONet, FNO, F-FNO, U-FNO, U-NO, LSM, MWT, Laplace NO, and state-space NO.
- **Geometry and transformer solvers:** Geo-FNO, GINO, GNOT, Transolver, UPT, MeshGraphNets, DoMINO, and ReViT.
- **CFD-specialized:** PIBERT, FourierFlow, P3D, AeroTransformer, NeuralDEM, DeepM&Mnet, and Energy Transformer.
- **Foundation and generative:** DPOT, Poseidon, PROSE-FD, BCAT, PDEformer-1, Tadpole, PDE-Refiner, FunDiff, and Flow Matching for PDEs.
- **Acceleration:** Solver-in-the-Loop, INC, PICT, diffSPH, NeuroSEM, neural-operator preconditioned Newton, and geometry-aware neural preconditioning.
- **Uncertainty and adaptation:** Conformalized-DeepONet and TANTE.

“Included” means represented through a uniform model card and recommendation interface. Official implementations remain external so upstream licenses and revisions are respected.

## Repository map

```text
src/navier_cfd/
  agents/          deterministic and LLM-ready planning
  benchmarks/      CFD metrics and benchmark plans
  datasets/        Hugging Face discovery, download, streaming
  models/          safe adapter protocol
  catalogs.py      model and dataset registries
  recommender.py   explainable task-to-model ranking

website/
  index.html       standalone project website
  recommender/     interactive browser recommender
  data/            static model and dataset catalogs

docs/              MkDocs technical documentation
case_studies/      detailed benchmark study guides
configs/tasks/     reusable task specifications
```

## Recommender validation

The Python recommender is covered by `pytest`; the browser engine is covered by Node's built-in test runner.

```bash
pytest tests/test_recommender.py
node --test website/recommender/recommender-core.test.mjs
```

Canonical tests verify that geometry-varying 3D tasks surface geometry-aware models, hybrid acceleration tasks surface solver-coupled methods, and CFD benchmark tasks surface relevant datasets.

## Packaging and deployment

- Package: `navier-cfd`
- Import namespace: `navier_cfd`
- Current version: `0.1.0`
- Build backend: Hatchling
- Website source: `website/`
- Documentation source: `docs/`
- GitHub Pages workflow: `.github/workflows/docs.yml`
- PyPI Trusted Publishing workflow: `.github/workflows/publish-pypi.yml`

The project website and documentation are deployed together as one dedicated NAVIER-CFD project site. GitHub Pages must be enabled once under **Settings → Pages → Source: GitHub Actions**.

## License

Licensed under the [Apache License 2.0](LICENSE). Cite NAVIER-CFD and the original model, dataset, upstream implementation, and numerical-solver references used in every experiment.
