# Unified experiments and PIBERT

NAVIER-CFD 0.4 introduces a reproducible execution layer connecting raw CFD records, canonical tensors, native or adapted models, training, checkpoints, and CFD-aware evaluation.

## Architecture

```text
Raw dataset record
        ↓
DatasetAdapter profile + explicit field mapping
        ↓
CFDSample / CFDBatch
        ↓
ModelBuildPlan
        ↓
ModelHub native or external constructor
        ↓
CFDTrainer
        ↓
CheckpointManager + CFD metric bundle
        ↓
Experiment manifest
```

## Canonical data contract

`CFDSample` uses channel-last arrays and supports both structured and unstructured data:

```python
from navier_cfd import CFDSample

sample = CFDSample(
    inputs=input_field,
    targets=target_field,
    coordinates=coordinates,
    parameters={"Re": 1000.0},
    mask=valid_domain,
    metadata={"case": "cylinder"},
)
```

Structured arrays may be `[nx, ny, channels]` or `[nx, ny, nz, channels]`. Point-cloud, mesh, and particle arrays use `[points, channels]`. The public collator stacks equal structured fields and pads variable point counts while producing a Boolean validity mask.

## Built-in dataset profiles

Profiles are provided for:

- PDEBench
- CFDBench
- RealPDEBench
- The Well
- APEBench
- ScalarFlow
- AirfRANS
- DrivAerNet++
- DrivAerML
- ShapeNet-Car
- EAGLE

Profiles recognize common aliases such as `input`, `history`, `state_in`, `target`, `future`, `solution`, `coords`, `grid`, `pos`, and `points`. Because dataset releases and local preprocessing pipelines differ, exact keys remain configurable:

```python
from navier_cfd import AdapterRegistry

adapter = AdapterRegistry().adapter(
    "cfdbench",
    input_key="history",
    target_key="future",
    coordinate_key="grid",
    mask_key="fluid_mask",
    input_fields=("u", "v", "p"),
    target_fields=("u", "v", "p"),
)
```

The adapter never silently invents a target. A missing required field raises `DatasetAdapterError`.

## Reproducible loaders

```python
from navier_cfd import AdaptedDataset, make_dataloaders

canonical = AdaptedDataset(raw_dataset, adapter)
loaders = make_dataloaders(
    canonical,
    batch_size=8,
    train=0.70,
    validation=0.15,
    test=0.15,
    seed=42,
    num_workers=4,
    pin_memory=True,
)
```

The split is deterministic for a fixed seed. Variable-size point-cloud samples are padded only along the point axis.

## Model configuration translation

```python
from navier_cfd import translate_model_config

plan = translate_model_config(
    "pibert",
    canonical[0],
    task=task,
    overrides={"hidden_dim": 256, "num_layers": 8},
)

print(plan.builder_kwargs)
print(plan.input_mode)
print(plan.notes)
```

Translation currently has explicit native rules for FNO, PIBERT, PINN, and DeepONet. External adapters receive a generic plan that must be reviewed against the upstream constructor.

## PIBERT

The native PIBERT reference implementation contains:

1. Input projection.
2. Multiscale Fourier coordinate features.
3. Sequence-local wavelet-detail features at configurable scales.
4. Multi-head self-attention with learnable coordinate-distance bias.
5. Optional PDE-residual attention bias.
6. Transformer residual blocks and a channel-wise output head.

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
    dropout=0.1,
)
```

Structured input:

```python
prediction = model(
    fields,       # [batch, nx, ny, input_dim]
    coordinates=grid,
    mask=fluid_mask,
)
```

Point input:

```python
prediction = model(
    point_features,       # [batch, points, input_dim]
    coordinates=points,
    mask=valid_points,
)
```

Optional PDE residuals may be supplied as `[batch, points]` or `[batch, points, residual_channels]` to focus attention on high-residual locations.

PIBERT attention is quadratic in the number of tokens. Large three-dimensional meshes should use downsampling, patches, latent tokens, or a model-specific sparse-attention extension rather than flattening millions of cells directly.

## Common training

```python
from navier_cfd import CFDTrainer, TrainerConfig

trainer = CFDTrainer(
    model,
    model_id="pibert",
    config=TrainerConfig(
        epochs=200,
        optimizer="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4,
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

Supported optimizers are Adam, AdamW, SGD, and LBFGS. Supported losses are MSE, L1/MAE, and Huber/Smooth-L1. A custom loss or forward function may be supplied for physics residuals, autoregressive rollout, closure coupling, or unusual external-model signatures.

## Checkpoints

```python
from navier_cfd import CheckpointManager

manager = CheckpointManager()
manager.save(
    "runs/checkpoint",
    model=model,
    optimizer=trainer.optimizer,
    scheduler=trainer.scheduler,
    config=trainer.config,
    metrics=metrics,
    metadata={"model_id": "pibert", "dataset_id": "pdebench"},
    epoch=100,
)
```

The checkpoint directory stores:

```text
weights.pt
optimizer.pt
scheduler.pt
manifest.json
```

The manifest records configuration, metrics, metadata, epoch, schema version, and timestamp.

## CFD metrics

The metric bundle includes:

- RMSE
- MAE
- normalized RMSE
- relative L2
- R²
- cosine similarity
- maximum absolute error
- spectral relative error
- RMS divergence for structured velocity fields
- kinetic-energy relative error
- rollout error by time step

Geometry-specific quantities such as drag, lift, pressure coefficient, wall shear, and integral flux require geometry, normals, areas, and reference values. These should be supplied through a benchmark-specific evaluator rather than inferred from fields alone.

## High-level Experiment API

```python
from navier_cfd import Experiment

experiment = Experiment(
    dataset_id="pdebench",
    model_id="pibert",
    task=task,
    trainer_config=config,
    batch_size=8,
    split_seed=42,
    output_dir="runs/pibert-pdebench",
)

result = experiment.run(raw_dataset)
```

The result contains training history, best validation loss, checkpoint path, metric bundle, resolved model build plan, and experiment manifest.

## External models

Every catalog model has a `ModelHandle`, but only models marked `native` or `external_adapter` are executable. External research repositories are not silently cloned. A validated adapter must identify:

- official repository and revision;
- license;
- installation specification;
- stable Python constructor;
- input/output tensor contract;
- checkpoint source and license;
- preprocessing and normalization;
- minimal construction and forward-pass test.

```python
from navier_cfd import ModelHub

hub = ModelHub()
hub.register_external(
    "transolver",
    entrypoint="reviewed_package:Transolver",
    install_spec="reviewed-package==x.y.z",
)
model = hub.load("transolver", hidden_dim=256)
```

This policy prevents catalog metadata from being misrepresented as runnable or scientifically equivalent code.
