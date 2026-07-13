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
  <img src="https://img.shields.io/badge/native_reference_models-52-008b8b.svg" alt="52 native reference models">
  <img src="https://img.shields.io/badge/dataset_profiles-11-5d8f72.svg" alt="11 dataset profiles">
</p>

NAVIER-CFD is a CFD-first Python platform for importing, configuring, training, testing, comparing, and recommending neural PDE/CFD models across standardized datasets. Version 0.5 makes the **dataset a first-class model-construction argument** and provides 52 executable native reference models under one common PyTorch workflow.

## Authors and affiliation

**Somyajit Chakraborty¹, Xizhong Chen¹\***

¹State Key Laboratory of Synergistic Chem-Bio Synthesis, Department of Chemical Engineering,  
School of Chemistry and Chemical Engineering, Shanghai Jiao Tong University, Shanghai 200240, China

**Correspondence**

- **Xizhong Chen（陈锡忠）:** [chenxizh@sjtu.edu.cn](mailto:chenxizh@sjtu.edu.cn)
- **Somyajit Chakraborty（叶一明）:** [chksomyajit@sjtu.edu.cn](mailto:chksomyajit@sjtu.edu.cn)

\* Corresponding author.

## Install

```bash
pip install "navier-cfd[models]"
```

Core catalog, recommendation, evidence, and dataset-discovery tools:

```bash
pip install navier-cfd
```

## Dataset determines the model configuration

```python
from navier_cfd import load_model

# 2D structured CFD configuration
fno = load_model("fno", dataset="cfdbench")

# 3D structured volumetric configuration
p3d = load_model("p3d", dataset="scalarflow")

# 2D point-cloud geometry configuration
transolver = load_model("transolver", dataset="airfrans")

# 3D unstructured geometry configuration
gino = load_model("gino", dataset="drivaerml")
```

A canonical sample overrides profile assumptions about channels, dimensionality, coordinates, and sensor count:

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

Resolution priority:

```text
explicit model keyword arguments
              ↑
user overrides
              ↑
actual CFDSample shapes
              ↑
registered dataset defaults
```

The build plan records:

- dataset identifier and representation;
- dimension and coordinate dimension;
- input and output channels;
- normalization description and layout;
- spectral modes or graph/attention parameters;
- model input mode;
- assumptions and scientific cautions.

## Registered dataset configurations

| Dataset | Representation | Default dimension | Primary use |
|---|---|---:|---|
| PDEBench | Structured | 2D default, sample-aware 1D–3D | PDE forecasting and operator learning |
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

Dataset defaults are visible and reviewable. Production experiments should pass an actual `CFDSample` and explicit adapter keys because historical releases can differ in variable names, ordering, resolution, and targets.

## Native reference model inventory

### Neural operators and operator learning

- DeepONet
- MIONet
- Fourier-DeepONet
- Nested Fourier-DeepONet
- Fourier-MIONet
- Fourier Neural Operator
- Physics-Informed Neural Operator
- Geo-FNO
- Geometry-Informed Neural Operator
- U-FNO
- Factorized FNO
- U-shaped Neural Operator
- Latent Spectral Model
- General Neural Operator Transformer
- Galerkin Transformer
- Multiwavelet Transformer
- FactFormer
- Orthogonal Neural Operator
- Transolver
- Laplace Neural Operator
- State-Space Neural Operator

### Physics-informed machine learning

- Physics-Informed Neural Network
- NSFnets
- PINNsFormer
- PINO
- PIBERT
- PI-MFM
- RiemannONet
- DeepM&Mnet

### Geometry, graph, transformer, and foundation-style learning

- MeshGraphNets
- DoMINO reference
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

### Generative, correction, preconditioning, adaptation, and uncertainty

- FourierFlow reference
- PDE-Refiner reference
- Solver-in-the-Loop corrector
- Indirect Neural Corrector
- NeuroSEM corrector reference
- Neural-operator preconditioned Newton reference
- Geometry-aware neural preconditioner
- Conformalized-DeepONet reference
- TANTE-style adaptation
- Energy Transformer reconstruction reference
- FunDiff reference
- Flow Matching for PDEs reference

The three remaining catalog entries—**PICT, diffSPH, and NeuralDEM**—remain specialized external integrations because meaningful execution requires their dedicated CFD/particle solver runtimes rather than a generic field-network substitute.

## Scientific scope

A NAVIER-CFD **native reference implementation** is:

- importable from the PyPI package;
- executable with PyTorch;
- connected to the dataset-aware configuration system;
- trainable with the common trainer;
- checkpointable;
- forward- and backward-tested;
- compatible with adapter conformance tests.

It is **not automatically**:

- a bit-for-bit copy of an author repository;
- a reproduction of unpublished preprocessing;
- a redistribution of private weights;
- numerically identical to every result in the originating paper.

Each reference model carries provenance and scope metadata such as:

```python
model.navier_reference_model_id
model.navier_reference_notice
model.navier_dataset_id
model.navier_build_plan
```

Numerical reproduction claims must still validate the official architecture, code revision, dataset split, preprocessing, losses, checkpoint, and evaluation protocol.

## Canonical dataset layer

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

The canonical `CFDSample`/`CFDBatch` layer supports:

- structured 1D, 2D, and 3D fields;
- point clouds and mesh nodes;
- variable-size samples;
- padding and validity masks;
- coordinates, parameters, and metadata;
- deterministic train/validation/test splits.

## Unified training

```python
from navier_cfd import CFDTrainer, TrainerConfig

trainer = CFDTrainer(
    model,
    model_id="transolver",
    config=TrainerConfig(
        epochs=200,
        optimizer="adamw",
        learning_rate=1e-3,
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

Supported optimizers and training features include Adam, AdamW, SGD, LBFGS, mixed precision, gradient clipping, cosine and plateau schedulers, early stopping, best checkpoints, periodic checkpoints, and custom forward/loss functions.

## High-level experiment API

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

Pipeline:

```text
raw dataset
   ↓
dataset adapter
   ↓
CFDSample / CFDBatch
   ↓
dataset-aware model configuration
   ↓
native or official external adapter
   ↓
common trainer
   ↓
checkpoint + CFD metrics + experiment manifest
```

## CFD evaluation

Built-in metrics include:

- RMSE and MAE;
- normalized RMSE;
- relative L2;
- R²;
- cosine similarity;
- maximum absolute error;
- spectral relative error;
- velocity-divergence RMS;
- kinetic-energy relative error;
- rollout error curves.

Drag, lift, pressure coefficient, wall shear, and other geometry-integrated quantities require case-specific surface normals, areas, reference quantities, and integration rules.

## Adapter conformance

```python
from navier_cfd import validate_model_adapter

report = validate_model_adapter(
    "gino",
    sample,
    dataset="airfrans",
)

print(report.to_dict())
```

Conformance checks registration, dependency availability, dataset-specific construction, parameter count, forward pass, output compatibility, and backward propagation.

## Evidence-aware recommendation

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

The recommender combines architecture compatibility with task-matched paper evidence, evidence quality, metric comparability, Bayesian shrinkage, confidence, and coverage reporting.

## Validation

```bash
pytest tests/test_model_hub.py tests/test_pibert_pipeline.py tests/test_native_suite.py
node --test website/recommender/recommender-core.test.mjs
```

The dedicated CPU-PyTorch CI job constructs, forwards, and backpropagates through all 52 native reference models and checks dataset-driven configuration for structured, 3D volumetric, point-cloud, and unstructured cases.

## Documentation

- Project website: https://samsomyajit.github.io/NAVIER-CFD/
- Interactive recommender: https://samsomyajit.github.io/NAVIER-CFD/recommender/
- Technical documentation: https://samsomyajit.github.io/NAVIER-CFD/docs/
- Simplified Chinese documentation: https://samsomyajit.github.io/NAVIER-CFD/docs/zh/
- Native suite guide: `docs/NATIVE_MODEL_SUITE.md`
- Unified experiments and PIBERT: `docs/UNIFIED_EXPERIMENTS.md`

## License

NAVIER-CFD is licensed under Apache-2.0. Original papers, repositories, datasets, model weights, and numerical solvers retain their own licenses and citation requirements.
