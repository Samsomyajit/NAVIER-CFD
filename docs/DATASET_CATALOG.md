# Dataset catalog and provider access

NAVIER-CFD registers 11 PDE/CFD dataset families as first-class objects. A dataset card records its physics, dimensions, representation, geometry and temporal modes, provider, split policy, and access contract.

| Dataset | Main role | Provider/access strategy |
|---|---|---|
| PDEBench | broad time-dependent PDE benchmark | selective Hugging Face HDF5 download plus NAVIER-CFD trajectory adapter |
| CFDBench | boundary, property, and geometry shifts | selective case ZIP from `chen-yingfa/CFDBench-raw` plus safe archive adapter |
| RealPDEBench | paired simulation and real measurements | selective scenario Arrow shards from `AI4Science-WestlakeU/RealPDEBench` |
| The Well | cross-physics simulation data and pretraining | official `the_well.data.WellDataset`; base `hf://datasets/polymathic-ai/` plus `well_dataset_name` |
| AirfRANS | 2D RANS geometry/OOD | upstream source and NAVIER-CFD adapter |
| APEBench | autoregressive differentiable benchmark | upstream generator and adapter |
| DrivAerNet++ / DrivAerML | realistic 3D aerodynamics | upstream sources and geometry adapters |
| ScalarFlow | real volumetric transport | upstream source and structured adapter |
| ShapeNet-Car | geometry design | upstream source and point-cloud adapter |
| EAGLE | fluid forecasting/reconstruction | upstream source and hybrid adapter |

## Why provider-aware access is necessary

A Hugging Face repository can contain Parquet, JSON, scientific HDF5, saved Arrow datasets, ZIP archives, or an official provider layout. Repository availability does not imply compatibility with `datasets.load_dataset(repo_id)`. NAVIER-CFD probes the storage layout and routes scientific formats through dedicated readers.

```bash
navier datasets auth-status
navier datasets probe pdebench --configuration burgers
navier datasets probe cfdbench --configuration cavity
navier datasets probe realpdebench --configuration cylinder
```

Credential precedence is:

```text
explicit token argument
    ↓
HF_TOKEN environment variable
    ↓
credential saved by `hf auth login`
    ↓
anonymous access
```

Tokens are never included in dataset access plans, experiment manifests, logs, checkpoints, or documentation output.

## Small PDEBench subset

```python
from navier_cfd import load_cfd_dataset

burgers = load_cfd_dataset(
    "pdebench",
    configuration="burgers",
    split="train",
    file_pattern="*.h5",
    trajectory_limit=32,
    max_windows=128,
    n_steps_input=4,
    n_steps_output=1,
)
```

PDEBench uses scientific HDF5 files. NAVIER-CFD downloads one selected file, opens it lazily, creates temporal windows, and returns canonical `CFDSample` objects. Pass an exact `filename`, `file_pattern`, and pinned `revision` for a reproducible experiment.

## Small CFDBench case

```python
cavity = load_cfd_dataset(
    "cfdbench",
    configuration="cavity",
    split="train",
    case=2,
    max_samples=64,
    temporal_pairs=True,
)
```

The loader downloads one selected scenario ZIP, validates extraction paths, and adapts supported NPZ, NPY, CSV, TXT, or DAT fields. Pickle payloads are not executed. The small-case split is deterministic within the selected archive and is not automatically equivalent to a paper's official case-level split.

## Small RealPDEBench subset

```python
cylinder = load_cfd_dataset(
    "realpdebench",
    configuration="cylinder",
    data_type="real",
    split="train",
    max_arrow_files=1,
    trajectory_limit=16,
    max_windows=64,
    n_steps_input=20,
    n_steps_output=20,
)
```

Small-compute mode selectively downloads Arrow shards and creates trajectory windows. Because it may load only part of a scenario, its train/validation/test partitions are explicitly marked as subset splits rather than official full-benchmark splits.

## Provider-native The Well access

```python
active_matter = load_cfd_dataset(
    "the_well",
    configuration="active_matter",
    split="train",
    streaming=True,
    n_steps_input=4,
    n_steps_output=1,
)
```

The Well remains on its official provider backend. When authentication is needed, the resolved credential is passed through provider storage options without being serialized.

## Reproducibility rules

- Pin the Hub revision whenever results will be reported.
- Record the selected repository, file or archive, configuration, and resolved revision.
- Preserve official splits where the provider exposes them.
- Label partial-file experiments as subset benchmarks.
- Never randomly place overlapping windows from one trajectory in both train and test sets.
- Record normalization, units, masks, boundaries, history length, prediction horizon, and split policy in the experiment manifest.
