<p align="center">
  <img src="docs/assets/navier-cfd-logo.svg" alt="NAVIER-CFD logo" width="860">
</p>

<h1 align="center">NAVIER-CFD</h1>

<p align="center"><strong>Neural and Agentic Verification, Integration, Evaluation, and Recommendation for Computational Fluid Dynamics</strong></p>

<p align="center">
  <a href="https://pypi.org/project/navier-cfd/"><img src="https://img.shields.io/pypi/v/navier-cfd.svg?label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/navier-cfd/"><img src="https://img.shields.io/pypi/pyversions/navier-cfd.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-4c8c6b.svg" alt="Apache 2.0"></a>
  <a href="https://github.com/Samsomyajit/NAVIER-CFD/actions/workflows/ci.yml"><img src="https://github.com/Samsomyajit/NAVIER-CFD/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://samsomyajit.github.io/NAVIER-CFD/"><img src="https://img.shields.io/badge/project-website-0d6fdc.svg" alt="Project website"></a>
  <a href="https://samsomyajit.github.io/NAVIER-CFD/recommender/"><img src="https://img.shields.io/badge/tool-recommender-13b7d8.svg" alt="Interactive recommender"></a>
  <img src="https://img.shields.io/badge/version-0.4.0-2f6f9f.svg" alt="Version 0.4.0">
  <img src="https://img.shields.io/badge/models-55-7c6aa6.svg" alt="55 models">
  <img src="https://img.shields.io/badge/native-PINN%20%7C%20DeepONet%20%7C%20FNO%20%7C%20PIBERT-008b8b.svg" alt="Native executable models">
  <img src="https://img.shields.io/badge/dataset_profiles-11-5d8f72.svg" alt="11 dataset profiles">
</p>

NAVIER-CFD is a CFD-first Python platform for executable neural PDE models, standardized dataset adaptation, reproducible training, checkpoints, CFD-aware metrics, evidence-aware recommendation, Hugging Face data access, and agentic experiment planning.

## Install

Core package:

```bash
pip install navier-cfd
```

Executable models and training:

```bash
pip install "navier-cfd[models]"
```

## What version 0.4 adds

- A canonical `CFDSample` and `CFDBatch` schema for structured grids, point clouds, unstructured meshes, and variable-size samples.
- Adapter profiles for PDEBench, CFDBench, RealPDEBench, The Well, APEBench, ScalarFlow, AirfRANS, DrivAerNet++, DrivAerML, ShapeNet-Car, and EAGLE.
- Reproducible train/validation/test splitting and PyTorch data loaders with padding and masks.
- Model-configuration translation from canonical samples.
- A common `CFDTrainer` with Adam, AdamW, SGD, LBFGS, mixed precision, schedulers, gradient clipping, early stopping, and checkpoints.
- Directory checkpoints containing weights, optimizer state, scheduler state, and a JSON manifest.
- Expanded CFD evaluation: RMSE, MAE, normalized RMSE, relative L2, R², cosine similarity, spectral error, divergence, kinetic-energy error, and rollout-error curves.
- A high-level `Experiment` API joining dataset adaptation, model construction, training, evaluation, and manifests.
- A native executable PIBERT reference implementation with Fourier coordinate embeddings, multiscale wavelet-detail embeddings, physics-biased attention, transformer blocks, and a field prediction head.

## One experiment API

```python
from navier_cfd import Experiment, TaskSpec, TrainerConfig

experiment = Experiment(
    dataset_id="pdebench",
    model_id="pibert",
    task=TaskSpec(
        problem="navier_stokes",
        task_type="forecasting",
        dimension=2,
        mesh_type="structured",
        temporal_mode="autoregressive",
        geometry_mode="fixed",
        physics=("incompressible_navier_stokes",),
    ),
    trainer_config=TrainerConfig(
        epochs=100,
        optimizer="adamw",
        learning_rate=1e-3,
        mixed_precision=True,
    ),
    batch_size=8,
    output_dir="runs/pibert-pdebench",
)

# raw_dataset may be a Hugging Face Dataset, list of mapping records,
# or another indexable dataset returning dictionaries.
result = experiment.run(raw_dataset)
print(result.metrics)
print(result.manifest_path)
```

Load a registered Hugging Face dataset first:

```python
raw_dataset = experiment.load_huggingface(split="train")
result = experiment.run(raw_dataset)
```

Dataset releases do not always use identical field names. The adapter can be made explicit:

```python
experiment.adapter_options = {
    "input_key": "history",
    "target_key": "future",
    "coordinate_key": "grid",
    "target_fields": ("u", "v", "p"),
}
```

## Direct PIBERT use

```python
from navier_cfd import load_model

model = load_model(
    "pibert",
    input_dim=4,
    output_dim=3,
    coordinate_dim=2,
    hidden_dim=128,
    num_layers=6,
    num_heads=8,
    num_frequencies=16,
    wavelet_scales=(1, 2, 4),
)

# Structured grid: [batch, nx, ny, channels]
prediction = model(fields, coordinates=grid)

# Point sequence: [batch, points, channels]
point_prediction = model(point_features, coordinates=point_coordinates, mask=point_mask)
```

The implementation is functional and tested, but it is a NAVIER-CFD reference implementation. Results should be validated against the exact architecture, preprocessing, losses, and splits described in the associated PIBERT study.

## Canonical dataset API

```python
from navier_cfd import AdaptedDataset, AdapterRegistry, make_dataloaders

adapter = AdapterRegistry().adapter(
    "airfrans",
    input_key="node_features",
    target_key="fields",
    coordinate_key="pos",
    input_fields=("normal_x", "normal_y", "distance"),
    target_fields=("pressure", "velocity_x", "velocity_y"),
)

dataset = AdaptedDataset(raw_airfrans, adapter)
loaders = make_dataloaders(
    dataset,
    batch_size=4,
    train=0.70,
    validation=0.15,
    test=0.15,
    seed=42,
)
```

Each canonical sample contains:

```text
inputs         structured field or [points, channels]
targets        target field or quantity
coordinates    optional grid, mesh, or point coordinates
parameters     Reynolds number, boundary values, controls, and other scalars
mask           optional valid-domain or valid-point mask
metadata       case identifiers and provenance
```

## Native executable models

| ID | Model | Status | Typical input |
|---|---|---|---|
| `pinn` | Physics-Informed Neural Network backbone | Native | Coordinates/parameters |
| `deeponet` | DeepONet | Native | Branch sensors + trunk coordinates |
| `fno` | 1D/2D/3D Fourier Neural Operator | Native | Structured fields |
| `pibert` | Fourier-wavelet physics-biased transformer | Native | Structured fields or point sequences |

```python
from navier_cfd import list_models

for handle in list_models():
    print(handle.id, handle.status.mode, handle.status.executable)
```

All 55 catalog models have a common `ModelHandle`. Four are currently implemented natively. Other models can be connected to reviewed upstream implementations through an explicit adapter:

```python
from navier_cfd import ModelHub

hub = ModelHub()
hub.register_external(
    "transolver",
    entrypoint="installed_transolver_package:Transolver",
    install_spec="installed-transolver-package",
)
model = hub.load("transolver", hidden_dim=256, num_layers=8)
```

NAVIER-CFD never silently clones or executes arbitrary research repositories. Exact adapters are added only when the upstream constructor, version, dependencies, license, tensor contract, and smoke test are known.

## Training and checkpoints

```python
from navier_cfd import CFDTrainer, TrainerConfig

trainer = CFDTrainer(
    model,
    model_id="pibert",
    config=TrainerConfig(
        epochs=200,
        optimizer="adamw",
        loss="mse",
        scheduler="cosine",
        gradient_clip=1.0,
        mixed_precision=True,
        checkpoint_dir="runs/checkpoints",
        checkpoint_every=25,
        early_stopping_patience=20,
    ),
)

training = trainer.fit(loaders["train"], loaders["validation"])
metrics = trainer.evaluate(loaders["test"], velocity=True)
```

Checkpoint layout:

```text
checkpoint/
  weights.pt
  optimizer.pt
  scheduler.pt
  manifest.json
```

## Evidence-aware recommendation

The recommender filters incompatible models and combines architecture compatibility with task-matched paper evidence. It reports final score, evidence score, confidence, coverage, supporting records, reasons, and cautions.

```bash
navier recommend \
  --problem vehicle_drag \
  --task surrogate \
  --dimension 3 \
  --mesh point_cloud \
  --temporal steady \
  --geometry varying \
  --physics aerodynamics \
  --fidelity rans \
  --memory-gb 80
```

Citation counts, venue prestige, and author prestige are not used as substitutes for CFD performance.

## Dataset profiles

| Dataset | Main representation | Default use |
|---|---|---|
| PDEBench | Structured | PDE surrogate and rollout |
| CFDBench | Structured | Boundary/property/geometry shifts |
| RealPDEBench | Structured | Simulation-to-real forecasting |
| The Well | Structured 2D/3D | Multiphysics pretraining and forecasting |
| APEBench | Structured 1D/2D/3D | Autoregressive emulation |
| ScalarFlow | Structured 3D | Scalar transport |
| AirfRANS | Point cloud/unstructured | Airfoil RANS |
| DrivAerNet++ | Surface point cloud | Vehicle aerodynamics |
| DrivAerML | Unstructured 3D | Vehicle CFD |
| ShapeNet-Car | Point cloud | Geometry-conditioned prediction |
| EAGLE | Structured/unstructured | Geometry-aware fluid learning |

Profiles provide field aliases and representation metadata. They are not a claim that every historical release of every dataset has the same schema. Use explicit adapter keys for the exact release being benchmarked.

## Project website

- Project site: https://samsomyajit.github.io/NAVIER-CFD/
- Interactive recommender: https://samsomyajit.github.io/NAVIER-CFD/recommender/
- Documentation: https://samsomyajit.github.io/NAVIER-CFD/docs/

## Validation

```bash
pytest
node --test website/recommender/recommender-core.test.mjs
```

CI covers Python 3.10–3.12, the evidence recommender, dataset adapters, browser runtime, model-hub behavior, native FNO/PINN/DeepONet construction, PIBERT structured and point-sequence forward passes, training, metrics, and checkpoint round trips.

## Scientific and licensing responsibility

A shared API does not make different model families scientifically interchangeable. Users must preserve and report:

- variable definitions and nondimensionalization;
- boundary and initial conditions;
- mesh/grid/point representation;
- temporal horizon and rollout protocol;
- training and validation splits;
- upstream model and checkpoint versions;
- dataset and model licenses;
- original model, dataset, and solver citations.

NAVIER-CFD is licensed under the [Apache License 2.0](LICENSE). Upstream implementations, pretrained weights, and datasets retain their own licenses.
