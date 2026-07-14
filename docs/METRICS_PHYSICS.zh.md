# 谱与物理导向指标

这些指标用于判断代理模型是否保留重要物理结构，而不仅仅是降低逐点误差。

## 谱相对误差

设 \(\mathcal F\) 为空间 Fourier 变换：

\[
E_{\mathrm{spec}}=
\frac{\|\,|\mathcal F(\hat y)|-|\mathcal F(y)|\,\|_2}
{\|\,|\mathcal F(y)|\,\|_2+\varepsilon}.
\]

它衡量整体 Fourier 幅值差异，但不能定位具体频段中的误差。

## 分频段谱 MSE

对波数区间 \(B\)：

\[
\mathrm{SMSE}(B)=
\frac{1}{N}
\sum_{\boldsymbol\omega\in B}
\left|\mathcal F(\hat y-y)(\boldsymbol\omega)\right|^2.
\]

NAVIER-CFD 默认使用正交归一化 FFT 和径向低/中/高频段。当所有频段覆盖完整频谱时，其总和在数值精度范围内等于空间域 MSE。

## Fourier RMSE（fRMSE）

按照 RealPDEBench 的解释，变换可以同时包含时间和空间。对频段 \(B\)：

\[
\mathrm{fRMSE}(B)=
\sqrt{
\frac{1}{K|B|}
\sum_{k=1}^{K}
\sum_{\boldsymbol\omega\in B}
\left\|\hat{\mathbf Y}_k(\boldsymbol\omega)-\mathbf Y_k(\boldsymbol\omega)\right\|_2^2
}.
\]

- 分别报告低频、中频和高频误差；
- 越低越好，\(0\) 为理想值；
- 必须记录 FFT 轴、归一化方式和频段边界。

## 时间频率误差

定义空间积分后的时间信号：

\[
s_k(t)=\sum_i y_k(t,\mathbf x_i).
\]

则：

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

FE 适用于涡脱落、周期反应器以及其他振荡动力学。

## 散度误差

对速度 \(\mathbf u=(u_1,\ldots,u_d)\)：

\[
\nabla\cdot\mathbf u=
\sum_{j=1}^{d}\frac{\partial u_j}{\partial x_j},
\qquad
D_{\mathrm{RMS}}=
\sqrt{\mathbb E[(\nabla\cdot\mathbf u)^2]}.
\]

NAVIER-CFD 报告：

\[
E_{\mathrm{div}}=
\left|D_{\mathrm{RMS}}(\hat{\mathbf u})-
D_{\mathrm{RMS}}(\mathbf u)\right|.
\]

该指标需要速度通道和网格间距信息，不能替代局部散度场或离散守恒检查。

## 动能相对误差

恒密度流动：

\[
e_k=\frac{1}{2}\sum_{j=1}^{d}u_j^2.
\]

变密度流动：

\[
e_k=\frac{1}{2}\rho\sum_{j=1}^{d}u_j^2.
\]

相对误差为：

\[
E_{\mathrm{KE}}=
\frac{\|\hat e_k-e_k\|_2}{\|e_k\|_2+\varepsilon}.
\]

## 湍动能误差

二维速度脉动的湍动能：

\[
k=\frac{1}{2}
\left(
\overline{(u-\bar u)^2}+
\overline{(v-\bar v)^2}
\right).
\]

NAVIER-CFD 计算预测与真实脉动动能场之间的平均绝对误差，对应 RealPDEBench 的长期统计一致性思想。

## 平均速度剖面误差

在探针点 \(\mathbf x_j\) 上：

\[
\mathrm{MVPE}=
\frac{1}{KN_p}
\sum_{k=1}^{K}\sum_{j=1}^{N_p}
\left|
\bar u_k(\mathbf x_j)-
\bar{\hat u}_k(\mathbf x_j)
\right|.
\]

如果没有给出探针，NAVIER-CFD 可以对非剖面方向进行平均并比较完整平均剖面。必须报告剖面轴和探针坐标。

## 涡量 RMSE

二维流动中：

\[
\omega=\frac{\partial v}{\partial x}-\frac{\partial u}{\partial y},
\qquad
E_\omega=
\sqrt{\mathbb E[(\hat\omega-\omega)^2]}.
\]

该指标对梯度和小尺度旋转结构较为敏感。

## 必需上下文

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

当输入信息不足时，指标返回 `valid=False` 和原因，而不会生成没有物理意义的数值。
