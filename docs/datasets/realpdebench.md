# RealPDEBench

Paired real measurements and matched simulations for sim-to-real forecasting.

**Registry ID:** `realpdebench`  
**Scenarios:** `cylinder`, `controlled_cylinder`, `fsi`, `foil`, `combustion`  
**Provider repository:** `AI4Science-WestlakeU/RealPDEBench`

## Inspect the provider

```bash
navier datasets auth-status
navier datasets probe realpdebench --configuration cylinder
```

## Small-compute loading

```python
from navier_cfd import load_cfd_dataset

train = load_cfd_dataset(
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

The Hub release stores complete trajectories in scenario-level Arrow shards and associated metadata/index files. NAVIER-CFD selectively downloads a limited number of Arrow shards, decodes the physical fields from the stored shape metadata, creates time windows, and returns canonical structured `CFDSample` objects.

Use `data_type="real"` or `data_type="numerical"`. For combustion data, numerical channels can be selected by index.

## Split and interpretation

A partial Arrow-shard run is explicitly marked `subset_mode=True` and `official_split=False`. It is suitable for small-machine portability and pipeline benchmarks, but it must not be presented as the full RealPDEBench official benchmark protocol.

Preserve numerical, real-only, fine-tuning, in-distribution, and OOD paradigms without leakage. Record the resolved revision, selected Arrow files, scenario, data type, trajectory identities, temporal window, field mapping, and split seed.
