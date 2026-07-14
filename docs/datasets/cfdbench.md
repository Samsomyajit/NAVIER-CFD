# CFDBench

CFD benchmark with varying boundary conditions, physical properties, and geometry.

**Registry ID:** `cfdbench`  
**Scenarios:** `cavity`, `tube`, `dam`, `cylinder`  
**Provider repository:** `chen-yingfa/CFDBench-raw`

## Inspect and select a case

```bash
navier datasets auth-status
navier datasets probe cfdbench --configuration cavity
navier datasets download cfdbench \
  --configuration cavity \
  --case 2 \
  --local-dir ./data/cfdbench
```

## Python access

```python
from navier_cfd import load_cfd_dataset

train = load_cfd_dataset(
    "cfdbench",
    configuration="cavity",
    split="train",
    case=2,
    max_samples=64,
    temporal_pairs=True,
)
```

The raw Hub repository stores per-scenario ZIP archives rather than a normal `datasets.load_dataset` table. NAVIER-CFD downloads only the selected case archive, validates extraction paths, and adapts supported NPZ, NPY, CSV, TXT, or DAT field files. Pickle payloads are never executed.

Depending on the selected archive, the adapter exposes point-wise coordinates and CFD target channels. Consecutive compatible frames can be converted into input–target temporal pairs.

## Reproducibility

Record the resolved Hub revision, exact archive, case identity, raw/interpolated representation, field mapping, units, nondimensionalization, time ordering, and split policy. A deterministic split made inside one downloaded archive is a small subset benchmark and must not be described as the paper's official case-level split unless independently verified.
