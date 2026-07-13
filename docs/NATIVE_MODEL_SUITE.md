# Dataset-configured native model suite

NAVIER-CFD 0.5 adds a dataset-aware import layer and a broad set of executable native reference architectures spanning operator learning, geometry-aware deep learning, transformer and graph models, and physics-informed machine learning.

## Dataset as an import argument

```python
from navier_cfd import load_model

model = load_model("fno", dataset="pdebench")
model = load_model("transolver", dataset="airfrans")
model = load_model("gino", dataset="drivaerml")
```

The dataset profile determines default:

- dimension;
- representation type;
- input and output channels;
- coordinate dimension;
- sensor count;
- Fourier modes;
- hidden width;
- number of layers;
- attention or graph dispatch;
- normalization description;
- channel layout.

A real canonical sample overrides dataset defaults:

```python
model, plan = load_model(
    "pibert",
    dataset="realpdebench",
    sample=sample,
    overrides={"hidden_dim": 256, "num_layers": 8},
    return_plan=True,
)

print(plan.to_dict())
```

The resolution order is:

1. explicit model keyword arguments;
2. user overrides;
3. actual `CFDSample` shapes;
4. registered dataset defaults.

## Dataset profiles

| Dataset | Representation | Default dimension | Typical model families |
|---|---|---:|---|
| PDEBench | structured | 2 | FNO, PINO, PIBERT, transformer operators |
| CFDBench | structured | 2 | FNO, PINO, U-NO, PIBERT |
| RealPDEBench | structured | 2 | PIBERT, PINO, DeepONet, temporal operators |
| The Well | structured | 3 | FNO, P3D, latent and state-space operators |
| APEBench | structured | 2 | autoregressive operators |
| ScalarFlow | structured | 3 | FNO, P3D, convolutional operators |
| AirfRANS | point cloud | 2 | Transolver, GINO, MeshGraphNets, Geo-FNO |
| DrivAerNet++ | point cloud | 3 | GINO, Transolver, AeroTransformer |
| DrivAerML | unstructured | 3 | GINO, MeshGraphNets, geometry transformers |
| ShapeNet-Car | point cloud | 3 | geometry-conditioned models |
| EAGLE | unstructured | 3 | graph and transformer operators |

Defaults are deliberately visible. Dataset releases can differ in field names, ordering, resolution, and target definitions, so production experiments should provide an actual sample and explicit adapter keys.

## Native reference inventory

### Operator learning

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

### Physics-informed ML

- PINN
- NSFnets
- PINNsFormer
- PINO
- PIBERT
- PI-MFM
- RiemannONet
- DeepM&Mnet

### Deep learning and geometry

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

## Native reference versus official reproduction

A NAVIER-CFD native reference implementation is:

- importable from the PyPI package;
- executable with PyTorch;
- connected to the common dataset API;
- trainable through `CFDTrainer`;
- checkpointable;
- forward- and backward-tested;
- covered by adapter conformance tests.

It is not automatically:

- the exact author repository revision;
- a reproduction of an unpublished preprocessing pipeline;
- a copy of private checkpoints;
- numerically identical to every paper table;
- licensed to redistribute third-party weights.

Each object includes a `navier_reference_notice` and should be benchmarked against the official source before reproduction claims are made.

## Conformance testing

```python
from navier_cfd import validate_model_adapter

report = validate_model_adapter(
    "transolver",
    sample,
    dataset="airfrans",
)

assert report.passed
```

The report checks:

- registry status;
- dependency availability;
- dataset-specific construction;
- parameter count;
- forward pass;
- output compatibility;
- backward propagation.

## Experiment API

```python
from navier_cfd import Experiment, TaskSpec, TrainerConfig

experiment = Experiment(
    dataset_id="cfdbench",
    model_id="pino",
    task=TaskSpec(
        problem="cylinder",
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
)

result = experiment.run(raw_dataset)
```

The experiment manifest records the dataset profile, actual sample-derived configuration, explicit overrides, model parameters, split seed, trainer settings, checkpoint path, and CFD metrics.
