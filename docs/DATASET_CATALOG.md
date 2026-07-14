# Dataset catalog and provider access

NAVIER-CFD registers 11 PDE/CFD dataset families as first-class objects. A dataset card records its physics, dimensions, representation, geometry and temporal modes, provider, official splits, and access contract.

| Dataset | Main role | Provider/access |
|---|---|---|
| PDEBench | broad time-dependent PDE benchmark | Hugging Face: `AI4Science-WestlakeU/PDEBench` |
| CFDBench | boundary, property, and geometry shifts | Hugging Face: `chen-yingfa/CFDBench` |
| RealPDEBench | paired simulation and real measurements | Hugging Face: `AI4Science-WestlakeU/RealPDEBench` |
| The Well | cross-physics simulation data and pretraining | official `the_well.data.WellDataset`; base `hf://datasets/polymathic-ai/` plus `well_dataset_name` |
| AirfRANS | 2D RANS geometry/OOD | upstream source and NAVIER-CFD adapter |
| APEBench | autoregressive differentiable benchmark | upstream generator and adapter |
| DrivAerNet++ / DrivAerML | realistic 3D aerodynamics | upstream sources and geometry adapters |
| ScalarFlow | real volumetric transport | upstream source and structured adapter |
| ShapeNet-Car | geometry design | upstream source and point-cloud adapter |
| EAGLE | fluid forecasting/reconstruction | upstream source and hybrid adapter |

## Generic Hugging Face access

```python
from navier_cfd.datasets import HuggingFaceDatasetManager

manager = HuggingFaceDatasetManager(token=None)
print(manager.discover("CFD fluid dynamics", limit=50))
manager.download("chen-yingfa/CFDBench", "./data/cfdbench", revision="<commit>")
stream = manager.load(
    "AI4Science-WestlakeU/RealPDEBench",
    split="train",
    streaming=True,
)
```

## Provider-native The Well access

The Well is not one `datasets.load_dataset` repository. Use the provider dispatcher and select an individual configuration:

```python
from navier_cfd import load_cfd_dataset

active_matter = load_cfd_dataset(
    "the_well",
    configuration="active_matter",
    split="train",
    streaming=True,
    n_steps_input=4,
    n_steps_output=1,
)
```

CLI download:

```bash
navier datasets well-list
navier datasets download the_well \
  --configuration active_matter \
  --split train \
  --local-dir ./data/the_well
```

## Split discipline

Preserve official provider splits whenever available. Do not randomly place overlapping windows from the same trajectory into train and test sets. Geometry generalization should be split by geometry identity; operating-condition generalization should be split by case or parameter range.

Always record provider version, dataset revision, configuration name, normalization statistics, units, masks, boundaries, history length, prediction horizon, and split policy in the experiment manifest.