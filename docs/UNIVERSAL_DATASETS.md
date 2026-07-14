# Universal dataset providers

NAVIER-CFD 0.8.0 gives every dataset in the built-in catalog a runtime loading path that returns canonical `CFDSample` objects. The access method follows the actual upstream storage and licensing model instead of pretending that every dataset is a normal Hugging Face table.

## Provider matrix

| Dataset | Runtime provider | Official access path | Small-compute controls |
|---|---|---|---|
| PDEBench | selective scientific HDF5 provider | selected files from the official PDEBench Hugging Face organization | filename/pattern, file-size cap, trajectory and window limits |
| CFDBench | selective case-archive provider | selected ZIP from `chen-yingfa/CFDBench-raw` | scenario, case, file-size cap, sample stride and limit |
| RealPDEBench | selective Arrow provider | selected official Arrow shards | scenario/type, shard cap, file-size cap, trajectory and window limits |
| The Well | official `WellDataset` provider | official API and storage options | individual dataset configuration, split, local cache and provider streaming |
| APEBench | official procedural provider | installed upstream scenario API | scenario group/kwargs, sample and window limits |
| AirfRANS | official archive plus local VTK provider | official processed or OpenFOAM archive | archive selection, transfer-size cap, split and sample limits |
| DrivAerNet++ | Dataverse/Globus plus local multimodal provider | selected Harvard Dataverse file ID, official URL, or official split metadata | selected artifact, size cap, split and sample limits |
| DrivAerML | official Hugging Face plus local mesh provider | selected `neashton/drivaerml` file | run/file path, revision, split and sample limits |
| ScalarFlow | official TUM FTP plus local volumetric provider | selected published FTP artifact | explicit remote file, size cap and temporal-window limits |
| ShapeNet-Car CFD | official direct archive plus local geometry provider | original `mlcfd_data.zip` distribution | transfer-size cap, extraction and sample limits |
| EAGLE | checksum-pinned direct archives plus local hybrid provider | selected official LIRIS archive | archive selection, SHA-256 verification and window limits |

## Official upstream staging

Use `OfficialDatasetManager` to inspect the publisher before transferring anything:

```python
from navier_cfd import OfficialDatasetManager

upstream = OfficialDatasetManager()
status = upstream.probe("eagle")
print(status.to_dict())
```

Stage only the official artifact needed for the experiment:

```python
result = upstream.download(
    "drivaerml",
    destination="/data/navier-official",
    artifacts=["run_1/boundary_1.vtp"],
)

print(result.files)
print(result.sha256)
print(result.manifest_path)
```

Then pass the staged file to the canonical loader:

```python
from navier_cfd import load_cfd_dataset

sample_set = load_cfd_dataset(
    "drivaerml",
    local_path=result.files[0],
    split="all",
    max_samples=1,
)
```

### Source-specific examples

AirfRANS processed archive:

```python
upstream.download(
    "airfrans",
    "/data/navier-official",
    artifacts=["processed"],
    max_bytes=50 * 1024**3,
    extract=True,
)
```

EAGLE checksum-pinned archive:

```python
upstream.download(
    "eagle",
    "/data/navier-official",
    artifacts=["step"],
    verify_checksum=True,
    extract=True,
)
```

ScalarFlow requires an explicit remote filename returned by `probe()` because individual reconstructions are several gigabytes:

```python
probe = upstream.probe("scalarflow")
print(probe.entries)

upstream.download(
    "scalarflow",
    "/data/navier-official",
    artifacts=["SELECTED_REMOTE_FILE"],
    max_bytes=8 * 1024**3,
)
```

For a selected DrivAerNet++ Harvard Dataverse CFD file:

```python
upstream.download(
    "drivaernetpp",
    "/data/navier-official",
    dataverse_file_id=12345678,
    max_bytes=20 * 1024**3,
)
```

The file ID must come from the official Dataverse record. NAVIER-CFD does not bypass license acceptance, registration, Globus authorization, or publisher access controls.

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

## Local canonical loading

After an official artifact is staged, `LocalScientificDatasetManager` or `load_cfd_dataset` reads supported NPZ, NPY, CSV, TXT, DAT, HDF5, VTK, VTP, VTU, PLY, STL, OpenFOAM marker files and safe PyTorch tensor exports.

```python
train = load_cfd_dataset(
    "airfrans",
    local_path="/data/navier-official/airfrans/extracted/Dataset",
    split="train",
    max_samples=32,
)
```

Temporal fields such as ScalarFlow and EAGLE are converted from time-first arrays into channel-last, time-flattened input and target tensors, matching the existing NAVIER-CFD structured-data contract.

## Security and provenance

- Upstream HTTP downloads are restricted to registered official hosts.
- Every staged file receives a SHA-256 digest and JSON provenance manifest.
- Published EAGLE checksums are verified after complete transfers.
- Transfer-size limits remove partial files when exceeded.
- ZIP and TAR members are checked against path traversal; archive links are rejected.
- NumPy files are opened with `allow_pickle=False`.
- PyTorch files require `weights_only=True`.
- Arbitrary pickle payloads are not executed.
- Credentials are never written to manifests.

## Validation grades

The live workflow uses explicit grades so a network probe is never confused with a parsed CFD sample:

| Grade | Meaning |
|---|---|
| `real_official_file_loaded` | publisher-hosted file downloaded, parsed into `CFDSample`, and checked for non-empty finite arrays |
| `official_procedural_sample_generated` | real installed upstream API generated a canonical trajectory |
| `official_file_downloaded` | genuine official file downloaded and validated, but the file is metadata rather than a CFD field |
| `official_endpoint_verified` | official endpoint or manifest reached without downloading the complete large collection |

Pull-request CI currently performs a real DrivAerML VTP download and parse, a real APEBench generation, official DrivAerNet++ split-file downloads, and live probes of the remaining public distribution channels. The scheduled workflow repeats these checks weekly so publisher-side file moves become visible.

Complete multi-gigabyte and multi-terabyte collections are not downloaded in pull-request CI. They can be selected explicitly through the same manager on the target workstation or cluster.
