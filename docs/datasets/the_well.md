# The Well

The Well is a provider family of large, diverse physics-simulation datasets. NAVIER-CFD integrates it as a first-class data provider while preserving NAVIER-CFD's broader purpose: multi-dataset integration, multi-family model construction, unified training, physical evaluation, evidence-aware recommendation, and chemical-engineering extensions.

## Correct access model

The Well is **not** one monolithic `datasets.load_dataset("polymathic-ai/the_well")` repository. Its official interface uses `the_well.data.WellDataset` with a provider base path and an individual dataset name:

```python
from navier_cfd import load_cfd_dataset

dataset = load_cfd_dataset(
    "the_well",
    configuration="active_matter",
    split="train",
    streaming=True,
    n_steps_input=4,
    n_steps_output=1,
    use_normalization=True,
)
```

Equivalent official provider arguments are:

```python
WellDataset(
    well_base_path="hf://datasets/polymathic-ai/",
    well_dataset_name="active_matter",
    well_split_name="train",
)
```

Install the optional provider package with:

```bash
pip install "navier-cfd[the-well]"
```

For local data, set `streaming=False` and pass `local_path`:

```python
dataset = load_cfd_dataset(
    "the_well",
    configuration="rayleigh_benard",
    split="valid",
    streaming=False,
    local_path="/data/the_well",
)
```

## Canonical adaptation

Official records use fields shaped as:

\[
[T, L_1, \ldots, L_d, F].
\]

By default, NAVIER-CFD converts them to:

\[
[L_1, \ldots, L_d, T F],
\]

so time history becomes model input channels while the physical spatial dimension remains correct. The adapter preserves:

- field names and tensor order;
- input and output time grids;
- spatial coordinates;
- constant scalars;
- boundary-condition metadata;
- official split;
- provider version and access plan;
- normalization provenance.

```python
sample = dataset[0]
model, plan = load_model(
    "fno",
    dataset="the_well",
    sample=sample,
    return_plan=True,
)
```

The actual sample determines the number of dimensions, fields, input-history channels, output channels, coordinates, and recommended model configuration.

## Official splits and normalization

Use The Well's official `train`, `valid`, and `test` splits. Do not randomly split overlapping windows from the same trajectories. The provider supports official Z-score or RMS normalization; physics metrics should normally be evaluated after restoring physical units.

## Scope statement

NAVIER-CFD uses The Well as a high-quality data and benchmark provider. It does not claim ownership of The Well datasets, reproduce the full 15 TB collection, or replace the official package. Cite The Well and the selected dataset source in addition to NAVIER-CFD.

## References

- Ohana et al., *The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning*, NeurIPS 2024.
- Official API: <https://polymathic-ai.org/the_well/api/>
- Official repository: <https://github.com/PolymathicAI/the_well>
