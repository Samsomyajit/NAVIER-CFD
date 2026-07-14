# Universal dataset providers

NAVIER-CFD 0.8.0 gives every dataset in the built-in catalog a runtime loading path that returns canonical `CFDSample` objects. The access method follows the actual upstream storage and licensing model instead of pretending that every dataset is a normal Hugging Face table.

## Provider matrix

| Dataset | Runtime provider | Access model | Small-compute controls |
|---|---|---|---|
| PDEBench | selective scientific HDF5 provider | downloads one matching HDF5 file from the selected PDEBench repository | filename/pattern, file-size cap, trajectory and window limits |
| CFDBench | selective case-archive provider | downloads one ZIP from `chen-yingfa/CFDBench-raw` and safely extracts supported fields | scenario, case, file-size cap, sample stride and limit |
| RealPDEBench | selective Arrow provider | downloads selected saved Arrow shards and decodes trajectory byte fields | scenario/type, shard cap, file-size cap, trajectory and window limits |
| The Well | official `WellDataset` provider | uses the official API and storage options | individual dataset configuration, split, local cache and provider streaming |
| APEBench | official procedural provider | calls the installed APEBench scenario API to generate deterministic trajectories | scenario group/kwargs, sample and window limits |
| AirfRANS | official-local VTK provider | reads `*_internal.vtu` with the optional matching `*_aerofoil.vtp` file | local subset, split manifest, stride and sample limit |
| DrivAerNet++ | licensed-local multimodal provider | reads selected locally staged VTK, HDF5, NPZ/NPY, text or safe tensor exports | file pattern, split manifest, stride and sample limit |
| DrivAerML | official-local mesh provider | reads selected VTK/OpenFOAM or exported scientific arrays | file pattern, split manifest, temporal horizons and limits |
| ScalarFlow | official-local volumetric provider | reads selected NPZ/NPY/HDF5 or exported tensor arrays | temporal history/horizon, stride and window limits |
| ShapeNet-Car | licensed-local geometry provider | reads locally staged geometry and companion CFD labels | file pattern, split manifest and sample limit |
| EAGLE | official-local hybrid provider | reads local mesh, array or tensor exports with safe tensor-only deserialization | temporal history/horizon, stride and window limits |

## Why some providers are local

AirfRANS, DrivAerNet++, DrivAerML, ScalarFlow, ShapeNet-Car and EAGLE are not treated as anonymous one-command downloads. Their official releases can be gated, licensed, manually distributed, extremely large, or stored in provider-specific layouts. NAVIER-CFD therefore separates two responsibilities:

1. The user obtains an authorized official copy or subset according to the upstream terms.
2. NAVIER-CFD probes that local export and converts supported files into the same `CFDSample` contract used by the trainer, models and metric suite.

This is universal runtime support, not a promise to bypass registration, licensing, storage, or access controls.

## Install optional readers

```bash
pip install "navier-cfd[scientific-data,mesh-data,torch]"
```

For procedural APEBench generation:

```bash
pip install "navier-cfd[apebench]"
```

## APEBench procedural generation

```python
from navier_cfd import load_cfd_dataset

advection = load_cfd_dataset(
    "apebench",
    configuration="Advection",
    scenario_group="difficulty",
    split="train",
    scenario_kwargs={"num_points": 128},
    n_steps_input=4,
    n_steps_output=1,
    max_samples=8,
    max_windows=64,
)
```

The provider records the installed APEBench version, scenario class, scenario name, generated split and subset policy in the access plan and sample metadata.

## AirfRANS local VTK loading

```python
from navier_cfd import load_cfd_dataset

train = load_cfd_dataset(
    "airfrans",
    local_path="/data/AirfRANS/Dataset",
    split="train",
    max_samples=32,
)
```

The provider searches for official `*_internal.vtu` case files and combines the matching `*_aerofoil.vtp` surface when present. Inputs include free-stream conditions, signed-distance information and surface normals; targets include velocity, pressure and turbulent viscosity fields when available.

## Automotive point-cloud subset

```python
cars = load_cfd_dataset(
    "drivaernetpp",
    local_path="/data/drivaernetpp_subset",
    split="train",
    file_pattern="**/*.npz",
    target_fields=("pressure", "wall_shear_stress"),
    max_samples=64,
)
```

The same local provider can read DrivAerML VTK/OpenFOAM exports. Split files named `train.txt`, `validation.txt`/`val.txt`, and `test.txt` are used when present; otherwise NAVIER-CFD creates a deterministic local subset split and marks it as non-official.

## ScalarFlow and EAGLE temporal windows

```python
scalar = load_cfd_dataset(
    "scalarflow",
    local_path="/data/scalarflow_subset",
    split="train",
    target_fields=("density", "velocity"),
    n_steps_input=4,
    n_steps_output=4,
    time_stride=1,
    window_stride=4,
    max_samples=4,
    max_windows=128,
)
```

Temporal fields are converted from time-first arrays into channel-last, time-flattened input and target tensors, matching the existing NAVIER-CFD structured-data contract.

## Probe before loading

```python
from navier_cfd import LocalScientificDatasetManager

manager = LocalScientificDatasetManager()
status = manager.probe(
    "eagle",
    local_path="/data/eagle_subset",
    file_pattern="**/*.pt",
)
print(status.to_dict())
```

The probe reports path availability, recognized file counts, detected formats and optional dependency availability without opening the complete dataset.

## Supported local formats and security

The local providers recognize NPZ, NPY, CSV, TXT, DAT, HDF5, VTK, VTP, VTU, PLY, STL, OpenFOAM marker files and PyTorch tensor exports.

Security rules are explicit:

- NumPy files are opened with `allow_pickle=False`.
- PyTorch files require `weights_only=True`.
- Arbitrary pickle payloads are not executed.
- Existing CFDBench ZIP extraction validates every destination path.
- Provider access plans record files and split provenance, never credentials.

## Validation scope

CI uses deterministic fixtures for procedural trajectories, structured time series and point-cloud records. It does not download complete upstream collections. A release is therefore validated at two levels:

- **Contract validation:** routing, parsing, canonical shapes, split provenance and security behavior are tested in CI.
- **Upstream validation:** real dataset subsets should still be probed and smoke-tested on the target machine because upstream files, licensing and storage layouts can change independently of NAVIER-CFD.
