# NAVIER-CFD

**Neural and Agentic Verification, Integration, Evaluation, and Recommendation for Computational Fluid Dynamics.**

NAVIER-CFD is a CFD-first Python platform for neural PDE solvers, hybrid acceleration, benchmark data, task-aware model recommendation, and agentic experiment planning.

## What makes it different

- CFD-first taxonomy spanning acceleration, surrogates, general PDE solvers, geometry models, specialized tasks, foundation models, inverse methods, uncertainty, and generative models.
- Uniform access to PDEBench, CFDBench, RealPDEBench, AirfRANS, DrivAer datasets, The Well, APEBench, ScalarFlow, and arbitrary Hugging Face datasets.
- Explainable model ranking by dimension, mesh, geometry, temporal regime, physics, memory, conservation, uncertainty, and numerical role.
- Deterministic and LLM-ready experiment planning.
- CFD-aware field, spectral, conservation, rollout, OOD, quantity-of-interest, and computational-cost metrics.
- Metadata-first external adapters that never execute untrusted repositories automatically.

## Install
```bash
pip install -e .
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

## Python API
```python
from navier_cfd import Catalog, TaskSpec, recommend_models

catalog = Catalog.load_builtin()
task = TaskSpec(
    problem="cylinder_wake",
    task_type="surrogate",
    dimension=2,
    mesh_type="structured",
    temporal_mode="autoregressive",
    geometry_mode="varying",
    physics=("incompressible_navier_stokes",),
    hardware_memory_gb=24,
)
for item in recommend_models(task, catalog.models, top_k=5):
    print(item.model.name, item.score, item.reasons, item.cautions)
```

See the [architecture](ARCHITECTURE.md), [model catalog](MODEL_CATALOG.md), [dataset catalog](DATASET_CATALOG.md), [agent layer](AGENTIC_AI.md), and [case studies](case_studies/README.md).
