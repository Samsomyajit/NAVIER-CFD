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
  <img src="https://img.shields.io/badge/version-1.1.0-2f6f9f.svg" alt="Version 1.1.0">
  <img src="https://img.shields.io/badge/catalog_models-55-7c6aa6.svg" alt="55 catalog models">
  <img src="https://img.shields.io/badge/native_reference_models-52-008b8b.svg" alt="52 native reference models">
  <img src="https://img.shields.io/badge/dataset_profiles-11-5d8f72.svg" alt="11 dataset profiles">
</p>

NAVIER-CFD is a CFD-first Python platform for importing, configuring, training, testing, comparing, and recommending neural PDE/CFD models across heterogeneous datasets.

```text
multi-dataset integration
+ multi-family model hub
+ dataset-conditioned construction
+ unified training and checkpoints
+ physical metric suites
+ evidence-aware recommendation
+ Codex-integrated AutoResearch
+ chemical-engineering extensions
```

Version 1.1.0 adds NAVIER AutoResearch: persistent research contracts, bounded agent sessions, Codex repository skills, a local read-only MCP server, deterministic CFD diagnostics, and auditable research-grade figure specifications.

## Authors and affiliation

**Somyajit Chakraborty¹, Kaiyuan Yang¹, Xizhong Chen¹\***

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

Official The Well provider support:

```bash
pip install "navier-cfd[models,the-well]"
```

Codex, MCP, and research-figure support:

```bash
pip install "navier-cfd[autoresearch]"
```

Core catalog, recommendation, evidence, dataset-discovery, and NumPy metric tools:

```bash
pip install navier-cfd
```

## NAVIER AutoResearch and Codex

Start an approval-aware campaign from a client problem:

```bash
navier-autoresearch init \
  "Reconstruct gas and solids velocities from EP_G history across unseen gas velocities" \
  --workspace runs/bubblenet-autoresearch \
  --domain gas_solid_multiphase \
  --mode guided \
  --max-gpu-hours 24 \
  --max-experiments 12
```

The campaign records a research contract, deterministic planner output, action proposals, approvals, findings, resource usage, and stopping decisions.

Codex integration is provided through:

- a repository `AGENTS.md` with scientific-integrity and autonomy rules;
- six skills under `.agents/skills`;
- a project MCP example under `.codex/config.toml.example`;
- the local `navier-autoresearch mcp` STDIO server.

The v1.1.0 MCP tools are intentionally read-only: dataset/model discovery, research planning, recommendation, metric catalog inspection, and figure-spec auditing. Training, solvers, cluster submission, large downloads, overwrites, and deletion are not exposed as automatic tools.

```python
from navier_cfd import AutoResearchSession, ResearchBudget, ResearchMode

session = AutoResearchSession.create(
    "runs/client-project",
    "Predict pressure drop and temperature fields for unseen heat-exchanger geometries",
    domain="heat_transfer",
    mode=ResearchMode.GUIDED,
    budget=ResearchBudget(max_gpu_hours=20, max_experiments=10),
)
plan = session.plan()
```

Detailed documentation:

- [AutoResearch overview](docs/AUTORESEARCH.md)
- [AutoResearch architecture](docs/AUTORESEARCH_ARCHITECTURE.md)
- [MCP tools](docs/AUTORESEARCH_TOOLS.md)
- [Codex skills](docs/CODEX_SKILLS.md)
- [Research contracts and sessions](docs/AUTORESEARCH_SESSIONS.md)
- [CFD diagnostics](docs/CFD_DIAGNOSTICS.md)
- [FigureLab](docs/FIGURELAB.md)

## Dataset-conditioned model construction

```python
from navier_cfd import load_model

fno = load_model("fno", dataset="cfdbench")
p3d = load_model("p3d", dataset="scalarflow")
transolver = load_model("transolver", dataset="airfrans")
gino = load_model("gino", dataset="drivaerml")
```

An actual canonical sample overrides static assumptions about dimensionality, coordinates, channels, history length, and sensor count:

```python
model, plan = load_model(
    "pibert",
    dataset="realpdebench",
    sample=sample,
    overrides={"hidden_dim": 256, "num_layers": 8, "num_heads": 8},
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

## Registered dataset families

| Dataset | Representation | Dimension | Primary use |
|---|---|---:|---|
| PDEBench | Structured | 1D–3D | PDE forecasting and operator learning |
| CFDBench | Structured | 2D | Cavity, tube, dam and cylinder CFD |
| RealPDEBench | Structured | 2D | Simulation-to-real forecasting |
| The Well | Structured provider family | 2D/3D | Multiphysics fields and pretraining |
| APEBench | Structured | 1D–3D | Autoregressive PDE emulation |
| ScalarFlow | Structured | 3D | Volumetric scalar transport |
| AirfRANS | Point cloud/unstructured | 2D | RANS airfoil geometry |
| DrivAerNet++ | Point cloud/unstructured | 3D | Vehicle aerodynamics |
| DrivAerML | Unstructured | 3D | High-fidelity vehicle CFD |
| ShapeNet-Car | Point cloud | 3D | Geometry-conditioned vehicle fields |
| EAGLE | Structured/unstructured | 2D/3D | Fluid and geometry learning |

## Official The Well provider

The Well is **not** treated as one fictitious `datasets.load_dataset("polymathic-ai/the_well")` repository. NAVIER-CFD calls its official `the_well.data.WellDataset` backend with:

```text
hf://datasets/polymathic-ai/
+ individual well_dataset_name
```

Load one configuration:

```python
from navier_cfd import load_cfd_dataset, load_model

dataset = load_cfd_dataset(
    "the_well",
    configuration="active_matter",
    split="train",
    streaming=True,
    n_steps_input=4,
    n_steps_output=1,
    use_normalization=True,
)

sample = dataset[0]
model = load_model("fno", dataset="the_well", sample=sample)
```

The adapter preserves:

- variable and constant field semantics;
- input and output time grids;
- spatial coordinates;
- scalar parameters;
- boundary-condition metadata;
- normalization type and statistics provenance;
- provider version and access plan;
- official `train`, `valid`, and `test` split identity.

Time history is flattened into channels by default, allowing the model builder to infer the physical spatial dimension and correct input/output widths.

## Native reference model coverage

NAVIER-CFD provides 52 executable reference implementations across:

- **operator learning:** DeepONet, MIONet, FNO/PINO, Geo-FNO, GINO, U-FNO, F-FNO, U-NO, LSM, GNOT, Galerkin Transformer, MWT, FactFormer, ONO, Transolver, Laplace NO, and State-Space NO;
- **physics-informed ML:** PINN, NSFnets, PINNsFormer, PIBERT, PI-MFM, RiemannONet, and DeepM&Mnet;
- **geometry and foundation models:** MeshGraphNets, DoMINO, UPT, DPOT, Poseidon, PROSE-FD, BCAT, PDEformer-1, P3D, AeroTransformer, Tadpole, and ReViT;
- **generative, correction, preconditioning, adaptation, and uncertainty:** FourierFlow, PDE-Refiner, solver-in-the-loop, INC, NeuroSEM, neural preconditioners, conformal prediction, TANTE-style adaptation, Energy Transformer, FunDiff, and flow matching.

PICT, diffSPH, and NeuralDEM remain specialized external integrations because meaningful execution requires their dedicated CFD or particle runtimes.

A NAVIER-CFD native reference implementation is executable, trainable, checkpointable, and forward/backward tested. It is not automatically a bit-for-bit reproduction of every author repository, unpublished preprocessing pipeline, private checkpoint, or paper table.

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
loaders = make_dataloaders(dataset, batch_size=4, seed=42)
```

`CFDSample` and `CFDBatch` support structured fields, point clouds, mesh nodes, variable-size samples, padding, masks, coordinates, physical parameters, and metadata.

## Unified training and evaluation

```python
from navier_cfd import CFDTrainer, MetricContext, TrainerConfig

metric_context = MetricContext(
    sample_axis=0,
    spatial_axes=(1, 2),
    channel_axis=-1,
    velocity_channels=(0, 1),
    spacing=(dx, dy),
)

trainer = CFDTrainer(
    model,
    model_id="transolver",
    config=TrainerConfig(
        epochs=200,
        optimizer="adamw",
        learning_rate=1e-3,
        mixed_precision=True,
        checkpoint_dir="runs/checkpoints",
        early_stopping_patience=20,
    ),
)

training = trainer.fit(loaders["train"], loaders["validation"])
metrics = trainer.evaluate(
    loaders["test"],
    metric_suites=("data_standard", "fluid_standard"),
    metric_context=metric_context,
)
```

The common trainer supports Adam, AdamW, SGD, LBFGS, mixed precision, gradient clipping, cosine and plateau schedulers, early stopping, periodic checkpoints, and custom forward/loss functions.

## Metric suites

```python
from navier_cfd import MetricContext, MetricSuite

context = MetricContext(
    sample_axis=0,
    time_axis=1,
    spatial_axes=(2, 3),
    channel_axis=-1,
    velocity_channels=(0, 1),
    spacing=(dx, dy),
    profile_axis=2,
)

suite = MetricSuite.combine([
    "data_standard",
    "the_well",
    "realpdebench",
    "fluid_standard",
])
results = suite.evaluate(prediction, target, context=context)
```

| Suite | Metrics |
|---|---|
| `data_standard` | MSE, RMSE, MAE, L∞, relative L1/L2, NMSE, NRMSE, R², Pearson, cosine |
| `the_well` | The Well-style normalized errors, VMSE, VRMSE, and binned spectral MSE |
| `realpdebench` | RMSE, MAE, per-sample relative L2, R², fRMSE, frequency error, turbulent KE, MVPE, Update Ratio |
| `fluid_standard` | RMSE, relative L2, spectral, divergence, kinetic-energy, and vorticity errors |

Every result records its category, optimization direction, ideal value, assumptions, validity, and evaluation space. Missing physical metadata returns `valid=False` with an explanation rather than a fabricated quantity.

## Research-grade figures and diagnostics

```python
from navier_cfd import FigureSpec, audit_figure_spec, analyze_interface_error

spec = FigureSpec(
    figure_type="truth_prediction_error",
    fields=("gas_velocity_y",),
    units="m/s",
    shared_color_limits=True,
    error_definition="absolute",
    mask="fluid_cells",
)
assert audit_figure_spec(spec).valid

interface_report = analyze_interface_error(
    prediction,
    target,
    gas_volume_fraction,
    spatial_axes=(1, 2),
)
```

The figure audit detects misleading color scales, missing units, undeclared masks, normalized fields mislabeled as physical quantities, visual smoothing, cherry-picking, low raster resolution, and missing vector output. Optional renderers create truth/prediction/error and profile figures with a `FigureManifest` sidecar.

## High-level experiment API

```python
from navier_cfd import Experiment, MetricContext, TaskSpec, TrainerConfig

experiment = Experiment(
    dataset_id="the_well",
    dataset_configuration="rayleigh_benard",
    model_id="pino",
    task=TaskSpec(
        problem="rayleigh_benard",
        task_type="forecasting",
        dimension=2,
        mesh_type="structured",
        temporal_mode="autoregressive",
        geometry_mode="fixed",
        physics=("incompressible_navier_stokes", "heat_transfer"),
    ),
    trainer_config=TrainerConfig(epochs=100, optimizer="adamw", mixed_precision=True),
    metric_suites=("data_standard", "the_well", "fluid_standard"),
    metric_context=MetricContext(velocity_channels=(0, 1), time_axis=1),
    output_dir="runs/pino-the-well",
)

# Preserve provider-native trajectories and avoid re-splitting overlapping windows.
splits = experiment.load_official_splits(streaming=True)
result = experiment.run(splits)
```

The experiment manifest records the provider, dataset configuration, split policy, model build plan, trainer configuration, checkpoints, metric definitions, context, assumptions, and results.

```text
provider or raw dataset
   ↓
canonical adapter + field semantics
   ↓
dataset-aware model construction
   ↓
common trainer and checkpoints
   ↓
physical metric suites
   ↓
auditable experiment manifest
```

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
pytest tests/test_the_well_provider.py tests/test_metric_suites.py
pytest tests/test_autoresearch.py tests/test_autoresearch_tools.py tests/test_figurelab.py
node --test website/recommender/recommender-core.test.mjs
mkdocs build --strict
```

CI verifies Python 3.10–3.12, all 52 native reference models, official provider contracts, dataset-conditioned model construction, metric analytical cases, AutoResearch contracts and tools, FigureLab audits, the browser recommender, and bilingual documentation.

## Documentation

- Project website: https://samsomyajit.github.io/NAVIER-CFD/
- Interactive recommender: https://samsomyajit.github.io/NAVIER-CFD/recommender/
- English documentation: https://samsomyajit.github.io/NAVIER-CFD/docs/
- Simplified Chinese documentation: https://samsomyajit.github.io/NAVIER-CFD/docs/zh/
- AutoResearch: `docs/AUTORESEARCH.md`
- Metrics: `docs/METRICS.md`
- The Well provider: `docs/datasets/the_well.md`
- Native model suite: `docs/NATIVE_MODEL_SUITE.md`

## Citation and scientific scope

When publishing results, cite NAVIER-CFD for the integration workflow, the original model paper, the original dataset/benchmark paper, and the official upstream implementation/checkpoint where applicable.

The Well datasets, RealPDEBench, model papers, repositories, checkpoints, and numerical solvers retain their own licenses and citation requirements.

## License

NAVIER-CFD is licensed under Apache-2.0.
