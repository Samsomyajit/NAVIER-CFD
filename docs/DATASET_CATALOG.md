# Dataset catalog and Hugging Face support

| Dataset | Main role | Hugging Face ID |
|---|---|---|
| PDEBench | broad time-dependent PDE benchmark | registered source/mirror |
| CFDBench | BC, property, and geometry shifts | `chen-yingfa/CFDBench` |
| RealPDEBench | paired simulation and real measurements | `AI4Science-WestlakeU/RealPDEBench` |
| The Well | cross-physics pretraining | `polymathic-ai/the_well` |
| AirfRANS | 2D RANS geometry/OOD | upstream source |
| APEBench | autoregressive differentiable benchmark | upstream generator |
| DrivAerNet++ / DrivAerML | realistic 3D aerodynamics | upstream sources |
| ScalarFlow | real volumetric transport | upstream source |
| ShapeNet-Car | geometry design | upstream source |
| EAGLE | fluid forecasting/reconstruction | registered source |

## Generic Hugging Face access
```python
from navier_cfd.datasets import HuggingFaceDatasetManager
manager = HuggingFaceDatasetManager(token=None)
print(manager.discover("CFD fluid dynamics", limit=50))
manager.download("chen-yingfa/CFDBench", "./data/cfdbench", revision="<commit>")
stream = manager.load("AI4Science-WestlakeU/RealPDEBench", split="train", streaming=True)
```

Always pin the dataset revision and preserve official OOD partitions.
