# PDEBench

Broad time-dependent PDE benchmark with classical and realistic equations.

**Registry ID:** `pdebench`  
**Provider:** scientific HDF5 repositories under the `pdebench` Hugging Face organization  
**Built-in configurations:** `burgers`, `advection`, `compressible_navier_stokes_1d`

## Inspect before downloading

```bash
navier datasets auth-status
navier datasets probe pdebench --configuration burgers
```

## Small-compute loading

```python
from navier_cfd import load_cfd_dataset

train = load_cfd_dataset(
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

The Hub repositories contain scientific trajectory HDF5 files rather than a conventional row-oriented `datasets.load_dataset` schema. NAVIER-CFD selectively downloads one file, opens it lazily, creates temporal windows, and returns canonical channel-last `CFDSample` objects.

## Reproducibility

Pin the Hub revision and record the exact file, HDF5 tensor key, trajectory split seed, history length, forecast horizon, stride, resolution, and physical variables. Small-compute splits are deterministic by trajectory and must be labelled as subset splits unless they reproduce an upstream official partition exactly.
