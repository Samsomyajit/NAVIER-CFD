# Spectral and physics-oriented metrics

These metrics evaluate whether a surrogate preserves physically important structure rather than only minimizing pointwise error.

## Spectral relative error

Let \(\mathcal F\) denote a spatial Fourier transform:

\[
E_{\mathrm{spec}}=
\frac{\|\,|\mathcal F(\hat y)|-|\mathcal F(y)|\,\|_2}
{\|\,|\mathcal F(y)|\,\|_2+\varepsilon}.
\]

This compares Fourier magnitudes globally. It does not localize errors to specific frequency bands.

## Binned spectral MSE

For a wavenumber bin \(B\):

\[
\mathrm{SMSE}(B)=
\frac{1}{N}
\sum_{\boldsymbol\omega\in B}
\left|\mathcal F(\hat y-y)(\boldsymbol\omega)\right|^2.
\]

NAVIER-CFD uses an orthonormal FFT and radial low/middle/high bins by default. When all bins cover the spectrum, their sum matches spatial-domain MSE up to numerical precision.

## Fourier RMSE (fRMSE)

Following the RealPDEBench interpretation, the transform can include time and space. For frequency bin \(B\):

\[
\mathrm{fRMSE}(B)=
\sqrt{
\frac{1}{K|B|}
\sum_{k=1}^{K}
\sum_{\boldsymbol\omega\in B}
\left\|\hat{\mathbf Y}_k(\boldsymbol\omega)-\mathbf Y_k(\boldsymbol\omega)\right\|_2^2
}.
\]

- Reports low-, middle-, and high-frequency errors.
- Lower is better; \(0\) is perfect.
- FFT axes, normalization and bin edges must be recorded.

## Temporal frequency error

Let the spatially integrated signal be:

\[
s_k(t)=\sum_i y_k(t,\mathbf x_i).
\]

Then:

\[
\mathrm{FE}=
\frac{1}{KT}
\sum_{k=1}^{K}
\sum_{\omega}
\left|
\mathcal F_t(s_k)(\omega)-
\mathcal F_t(\hat s_k)(\omega)
\right|.
\]

FE is useful for vortex shedding, oscillatory reactors and other periodic dynamics.

## Divergence error

For velocity \(\mathbf u=(u_1,\ldots,u_d)\):

\[
\nabla\cdot\mathbf u=
\sum_{j=1}^{d}\frac{\partial u_j}{\partial x_j},
\qquad
D_{\mathrm{RMS}}=
\sqrt{\mathbb E[(\nabla\cdot\mathbf u)^2]}.
\]

NAVIER-CFD reports:

\[
E_{\mathrm{div}}=
\left|D_{\mathrm{RMS}}(\hat{\mathbf u})-
D_{\mathrm{RMS}}(\mathbf u)\right|.
\]

This requires velocity-channel and grid-spacing metadata. It is not a substitute for checking local divergence fields or discrete solver conservation.

## Kinetic-energy relative error

For constant-density flow:

\[
e_k=\frac{1}{2}\sum_{j=1}^{d}u_j^2.
\]

For variable density:

\[
e_k=\frac{1}{2}\rho\sum_{j=1}^{d}u_j^2.
\]

The relative field error is:

\[
E_{\mathrm{KE}}=
\frac{\|\hat e_k-e_k\|_2}{\|e_k\|_2+\varepsilon}.
\]

## Turbulent kinetic-energy error

For two-dimensional velocity fluctuations:

\[
k=\frac{1}{2}
\left(
\overline{(u-\bar u)^2}+
\overline{(v-\bar v)^2}
\right).
\]

NAVIER-CFD computes the mean absolute difference between predicted and target fluctuation-energy fields. This follows the RealPDEBench-style long-time consistency interpretation.

## Mean velocity profile error

At selected probes \(\mathbf x_j\):

\[
\mathrm{MVPE}=
\frac{1}{KN_p}
\sum_{k=1}^{K}\sum_{j=1}^{N_p}
\left|
\bar u_k(\mathbf x_j)-
\bar{\hat u}_k(\mathbf x_j)
\right|.
\]

When probes are not supplied, NAVIER-CFD can average over all non-profile spatial axes and compare full mean profiles. The selected profile axis and probe coordinates must be reported.

## Vorticity RMSE

For two-dimensional flow:

\[
\omega=\frac{\partial v}{\partial x}-\frac{\partial u}{\partial y},
\qquad
E_\omega=
\sqrt{\mathbb E[(\hat\omega-\omega)^2]}.
\]

This metric is sensitive to gradients and small-scale rotational structure.

## Required context

```python
context = MetricContext(
    time_axis=1,
    spatial_axes=(2, 3),
    channel_axis=-1,
    velocity_channels=(0, 1),
    spacing=(dx, dy),
    profile_axis=2,
    evaluation_space="physical",
)
```

Metrics that cannot be supported by the supplied context are returned with `valid=False` and an explanation.
