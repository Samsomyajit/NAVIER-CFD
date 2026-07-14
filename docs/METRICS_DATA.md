# Data-oriented metrics

Let \(y_i\) denote target values, \(\hat y_i\) predictions, and \(N\) the number of evaluated values after masking.

## Mean squared error

\[
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2.
\]

- Range: \([0,\infty)\)
- Best value: \(0\)
- Lower is better.

## Root mean squared error

\[
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2}.
\]

RMSE has the same units as the predicted field and emphasizes large errors.

## Mean absolute error

\[
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|\hat y_i-y_i|.
\]

MAE is less sensitive to isolated large errors than RMSE.

## Maximum absolute error

\[
L_\infty=\max_i|\hat y_i-y_i|.
\]

This reports the worst evaluated point.

## Relative \(L_1\) and \(L_2\)

\[
\mathrm{RelL1}=\frac{\|\hat{\mathbf y}-\mathbf y\|_1}{\|\mathbf y\|_1+\varepsilon},
\qquad
\mathrm{RelL2}=\frac{\|\hat{\mathbf y}-\mathbf y\|_2}{\|\mathbf y\|_2+\varepsilon}.
\]

The `realpdebench` suite uses the mean of the per-sample relative \(L_2\) values:

\[
\frac{1}{K}\sum_{k=1}^{K}
\frac{\|\hat{\mathbf y}_k-\mathbf y_k\|_2}{\|\mathbf y_k\|_2+\varepsilon}.
\]

## NMSE and NRMSE

\[
\mathrm{NMSE}=
\frac{\mathbb E[(\hat y-y)^2]}{\mathbb E[y^2]+\varepsilon},
\qquad
\mathrm{NRMSE}=\sqrt{\mathrm{NMSE}}.
\]

These are scale-normalized, but the exact normalization must be stated when comparing external benchmarks.

## Coefficient of determination

\[
R^2=1-
\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y)^2+\varepsilon}.
\]

- Best value: \(1\)
- Higher is better.
- Values below zero indicate that the predictor is worse than the target mean under this aggregation.

## Pearson correlation

\[
r=
\frac{(\hat{\mathbf y}-\bar{\hat y})\cdot(\mathbf y-\bar y)}
{\|\hat{\mathbf y}-\bar{\hat y}\|_2\,\|\mathbf y-\bar y\|_2+\varepsilon}.
\]

It measures linear association and does not by itself measure calibration or magnitude accuracy.

## Cosine similarity

\[
\mathrm{CosSim}=
\frac{\hat{\mathbf y}\cdot\mathbf y}
{\|\hat{\mathbf y}\|_2\|\mathbf y\|_2+\varepsilon}.
\]

Useful for profile or directional agreement, but insensitive to uniform amplitude scaling.

## Variance-scaled errors

For field/channel \(c\) with variance \(\sigma_c^2\):

\[
\mathrm{VMSE}=
\frac{1}{C}\sum_{c=1}^{C}
\frac{\mathrm{MSE}_c}{\sigma_c^2+\varepsilon},
\qquad
\mathrm{VRMSE}=\sqrt{\mathrm{VMSE}}.
\]

NAVIER-CFD uses provider-supplied field variance when available; otherwise it estimates variance from the target array. Record which source was used.

## Update Ratio

For simulated pretraining followed by real-data finetuning:

\[
\mathrm{Update\ Ratio}=\frac{N_1}{N_2},
\]

where \(N_1\) is the number of finetuning updates and \(N_2\) the number of scratch-training updates required to reach the best scratch-training RMSE.

- Lower is better.
- Values below \(1\) indicate that pretraining reduced the number of updates.
- This metric is valid only when the target performance threshold and update-count protocol are identical.

## API

```python
from navier_cfd import MetricContext, MetricSuite

results = MetricSuite.from_name("data_standard").evaluate(
    prediction,
    target,
    context=MetricContext(mask=fluid_mask),
)
```
