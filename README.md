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
  <img src="https://img.shields.io/badge/version-0.5.0-2f6f9f.svg" alt="Version 0.5.0">
  <img src="https://img.shields.io/badge/catalog_models-55-7c6aa6.svg" alt="55 catalog models">
  <img src="https://img.shields.io/badge/native_reference_models-39-008b8b.svg" alt="39 native reference models">
  <img src="https://img.shields.io/badge/dataset_profiles-11-5d8f72.svg" alt="11 dataset profiles">
</p>

NAVIER-CFD is a CFD-first Python platform for importing, configuring, training, testing, and comparing neural PDE/CFD models across standardized datasets. It combines dataset adapters, native reference architectures, external-model adapters, evidence-aware recommendation, reproducible training, checkpoints, CFD metrics, Hugging Face access, and experiment manifests.

## Install

```bash
pip install "navier-cfd[models]"
```

Core catalog and recommender only:

```bash
pip install navier-cfd
```

## Dataset-driven model import

The dataset can be passed directly to `load_model`. NAVIER-CFD resolves dimensionality, field channels, coordinate size, mesh representation, spectral modes, hidden size, and input dispatch from the registered dataset profile.

```python
from navier_cfd import load_model

# 2D structured CFD defaults: 3 input and 3 output channels.
fno = load_model("fno", dataset="cfdbench")

# 3D structured scalar-flow defaults and 3D Fourier modes.
fno_3d = load_model("fno", dataset="scalarflow")

# Point-cloud geometry defaults for AirfRANS.
transolver = load_model("transolver", dataset="airfrans")

# Unstructured 3D defaults for DrivAerML.
gino = load_model("gino", dataset="drivaerml")
```

A real canonical sample overrides profile assumptions:

```python
model, plan = load_model(
    "pibert",
    dataset="realpdebench",
    sample=sample,
    overrides={
        "hidden_dim": 256,
        "num_layers": 8,
        "num_heads": 8,
    },
    return_plan=True,
)

print(plan.to_dict())
```

Resolution priority is:

```text
explicit keyword arguments
        ↑
user overrides
        ↑
actual CFDSample shapes
        ↑
registered dataset defaults
```

## Registered dataset configurations

| Dataset | Default representation | Dimension | Typical use |
|---|---|---:|---|
| PDEBench | Structured | 2D default; sample-aware 1D–3D | PDE forecasting and operators |
| CFDBench | Structured | 2D | Cavity, tube, dam and cylinder CFD |
| RealPDEBench | Structured | 2D | Simulation-to-real forecasting |
| The Well | Structured | 3D default | Large multiphysics fields |
| APEBench | Structured | 2D default | Autoregressive PDE emulation |
| ScalarFlow | Structured | 3D | Volumetric scalar transport |
| AirfRANS | Point cloud | 2D | RANS airfoil geometry |
| DrivAerNet++ | Point cloud | 3D | Vehicle aerodynamics |
| DrivAerML | Unstructured | 3D | High-fidelity vehicle CFD |
| ShapeNet-Car | Point cloud | 3D | Geometry-conditioned vehicle fields |
| EAGLE | Unstructured | 3D default | Fluid and geometry learning |

Dataset defaults are reviewable, not hidden. Historical releases may use different variable names or channel order, so production experiments should pass an actual `CFDSample` and explicit adapter keys.

## Native reference model suite

Version 0.5 exposes **39 executable native reference models** through one interface.

### Neural operators and operator-learning models

- DeepONet
- MIONet
- Fourier-DeepONet
- Nested Fourier-DeepONet
- Fourier-MIONet
- FNO
- PINO
- Geo-FNO
- GINO
- U-FNO
- F-FNO
- U-NO
- Latent Spectral Model
- GNOT
- Galerkin Transformer
- Multiwavelet Transformer
- FactFormer
- Orthogonal Neural Operator
- Transolver
- Laplace Neural Operator
- State-Space Neural Operator

### Physics-informed models

- PINN
- NSFnets
- PINNsFormer
- PINO
- PIBERT
- PI-MFM
- RiemannONet
- DeepM&Mnet

### Deep-learning, geometry, and foundation-style models

- MeshGraphNets
- Universal Physics Transformer
- DPOT
- Poseidon
- PROSE-FD
- BCAT
- PDEformer-1
- P3D
- AeroTransformer
- Tadpole
- ReViT

Inspect availability:

```python
from navier_cfd import list_models

for handle in list_models():
    print(handle.id, handle.status.mode, handle.status.executable)
```

## Scientific scope of native models

A **native reference implementation** is executable, trainable, shape-tested, backward-tested, and connected to the common dataset/trainer API. It is not automatically a bit-for-bit reproduction of an author repository, unpublished preprocessing pipeline, private checkpoint, or paper table.

Every native reference object carries:

```python
model.navier_reference_model_id
model.navier_reference_notice
model.navier_dataset_id
model.navier_build_plan
```

Numerical claims must still cite and validate the original method, official implementation, dataset split, preprocessing, loss, and checkpoint.

## Unified dataset and experiment API

```python
from navier_cfd import Experiment, TaskSpec, TrainerConfig

experiment = Experiment(
    dataset_id="pdebench",
    model_id="pino",
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
    output_dir="runs/pino-pdebench",
)

result = experiment.run(raw_dataset)
print(result.metrics)
print(result.build_plan)
print(result.manifest_path)
```

The pipeline is:

```text
raw dataset
   ↓
dataset adapter
   ↓
CFDSample / CFDBatch
   ↓
dataset-aware model configuration
   ↓
native or external model adapter
   ↓
common trainer
   ↓
checkpoint + CFD metrics + experiment manifest
```

## Canonical data layer

```python
from navier_cfd import AdaptedDataset, AdapterRegistry, make_dataloaders

adapter = AdapterRegistry().adapter(
    "airfrans",
    input_key="node_features",
    target_key="fields",
    coordinate_key="pos",
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

The loaders support structured grids, point clouds, mesh nodes, variable-size samples, padding, validity masks, deterministic splits, and parameter dictionaries.

## Training and checkpoints

```python
from navier_cfd import CFDTrainer, TrainerConfig

trainer = CFDTrainer(
    model,
    model_id="transolver",
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

Checkpoint directories contain weights, optimizer state, scheduler state, metrics, configuration, and provenance metadata.

## CFD evaluation

The evaluation layer includes:

- RMSE and MAE
- normalized RMSE
- relative L2
- R²
- cosine similarity
- maximum absolute error
- spectral relative error
- divergence RMS
- kinetic-energy relative error
- rollout error curves

Geometry-dependent drag, lift, pressure coefficient, and wall-shear evaluation require case-specific surface normals, areas, reference quantities, and integration rules.

## Adapter conformance testing

```python
from navier_cfd import validate_model_adapter

report = validate_model_adapter(
    "gino",
    sample,
    dataset="airfrans",
)

print(report.to_dict())
```

Conformance checks registration, dependency availability, dataset-specific construction, forward pass, shape compatibility, and backward propagation.

## Evidence-aware recommendation

The recommender combines hard compatibility filtering with task-matched paper evidence, evidence quality, metric comparability, Bayesian shrinkage, confidence, and coverage reporting.

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

## External official implementations

Models without a native reference can still be connected through an explicit installed entrypoint:

```python
from navier_cfd import ModelHub

hub = ModelHub()
hub.register_external(
    "domino",
    entrypoint="installed_domino_package:DoMINO",
    install_spec="installed-domino-package",
)
model = hub.load("domino", hidden_dim=256)
```

NAVIER-CFD never silently clones or executes arbitrary repositories. External adapters require a reviewed constructor, version, dependency set, tensor contract, license, and smoke test.

## Validation

```bash
pytest tests/test_model_hub.py tests/test_pibert_pipeline.py tests/test_native_suite.py
node --test website/recommender/recommender-core.test.mjs
```

The native CI job constructs, forwards, and backpropagates through the dataset-configured native suite using CPU PyTorch.

## License

NAVIER-CFD is licensed under Apache-2.0. Original models, datasets, code repositories, and checkpoints retain their own licenses and citation requirements.
