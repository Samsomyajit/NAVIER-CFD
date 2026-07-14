# Evaluation metrics

NAVIER-CFD 0.6 introduces named metric suites for data accuracy, spectral fidelity, physical consistency, and training efficiency. Metrics are returned with their direction, ideal value, assumptions, validity, and evaluation-space metadata.

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
    "fluid_standard",
])
results = suite.evaluate(prediction, target, context=context)
```

## Built-in suites

| Suite | Purpose | Main metrics |
|---|---|---|
| `data_standard` | General field accuracy | MSE, RMSE, MAE, L∞, relative L1/L2, NMSE, NRMSE, R², Pearson, cosine |
| `the_well` | The Well-compatible normalized and spectral evaluation | L∞, MAE, MSE, RMSE, NMSE, NRMSE, Pearson, VMSE, VRMSE, binned spectral MSE |
| `realpdebench` | RealPDEBench-style data and physics evaluation | RMSE, MAE, per-sample relative L2, R², fRMSE, FE, turbulent KE, MVPE, Update Ratio |
| `fluid_standard` | CFD physical consistency | RMSE, relative L2, spectral relative error, divergence error, kinetic-energy error, vorticity RMSE |

## Metric result contract

Every metric returns a `MetricResult`:

```python
{
    "name": "kinetic_energy_relative_error",
    "value": 0.031,
    "category": "physics",
    "direction": "lower",
    "best_value": 0.0,
    "valid": True,
    "assumptions": ["velocity channels are declared"],
    "metadata": {"evaluation_space": "physical"},
}
```

A metric that lacks required metadata is marked invalid rather than fabricated. For example, drag cannot be computed without surface normals, areas, pressure/shear fields, and reference quantities.

## Evaluation space

Data metrics may be reported in normalized space when stated explicitly. Physics metrics should normally be evaluated after inverse normalization in physical units. Record the normalization method, statistics source, mask policy, axes, channel mapping, and aggregation in the experiment manifest.

## Sources and compatibility

The metric package adopts useful ideas from The Well and RealPDEBench while keeping explicit compatibility labels. Similar names do not guarantee identical normalization, FFT convention, binning, probe placement, or aggregation. Reproduction studies must record those details and cite the original benchmark.

- The Well API: <https://polymathic-ai.org/the_well/api/>
- RealPDEBench data-oriented metrics: <https://realpdebench.github.io/metrics/data-oriented/>
- RealPDEBench physics-oriented metrics: <https://realpdebench.github.io/metrics/physics-oriented/>
