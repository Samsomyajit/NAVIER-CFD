# 数据导向指标

设 \(y_i\) 为真实值、\(\hat y_i\) 为预测值，\(N\) 为掩码后参与评价的数值总数。

## 均方误差

\[
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2.
\]

- 范围：\([0,\infty)\)
- 最优值：\(0\)
- 越低越好。

## 均方根误差

\[
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2}.
\]

RMSE 与预测物理量具有相同单位，并且对较大误差更加敏感。

## 平均绝对误差

\[
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|\hat y_i-y_i|.
\]

与 RMSE 相比，MAE 对少量极端误差不那么敏感。

## 最大绝对误差

\[
L_\infty=\max_i|\hat y_i-y_i|.
\]

该指标反映评价区域中的最坏点误差。

## 相对 \(L_1\) 与 \(L_2\)

\[
\mathrm{RelL1}=\frac{\|\hat{\mathbf y}-\mathbf y\|_1}{\|\mathbf y\|_1+\varepsilon},
\qquad
\mathrm{RelL2}=\frac{\|\hat{\mathbf y}-\mathbf y\|_2}{\|\mathbf y\|_2+\varepsilon}.
\]

`realpdebench` 套件采用逐样本相对 \(L_2\) 的平均值：

\[
\frac{1}{K}\sum_{k=1}^{K}
\frac{\|\hat{\mathbf y}_k-\mathbf y_k\|_2}{\|\mathbf y_k\|_2+\varepsilon}.
\]

## NMSE 与 NRMSE

\[
\mathrm{NMSE}=
\frac{\mathbb E[(\hat y-y)^2]}{\mathbb E[y^2]+\varepsilon},
\qquad
\mathrm{NRMSE}=\sqrt{\mathrm{NMSE}}.
\]

这些指标消除了部分尺度影响，但与外部基准比较时必须说明具体归一化定义。

## 决定系数

\[
R^2=1-
\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y)^2+\varepsilon}.
\]

- 最优值：\(1\)
- 越高越好。
- 小于零表示在当前聚合方式下，模型比直接预测真实场均值更差。

## Pearson 相关系数

\[
r=
\frac{(\hat{\mathbf y}-\bar{\hat y})\cdot(\mathbf y-\bar y)}
{\|\hat{\mathbf y}-\bar{\hat y}\|_2\,\|\mathbf y-\bar y\|_2+\varepsilon}.
\]

它衡量线性相关性，但不能单独反映幅值精度或校准质量。

## 余弦相似度

\[
\mathrm{CosSim}=
\frac{\hat{\mathbf y}\cdot\mathbf y}
{\|\hat{\mathbf y}\|_2\|\mathbf y\|_2+\varepsilon}.
\]

适合衡量剖面形状或方向一致性，但对统一幅值缩放不敏感。

## 方差缩放误差

对场或通道 \(c\)，其方差为 \(\sigma_c^2\)：

\[
\mathrm{VMSE}=
\frac{1}{C}\sum_{c=1}^{C}
\frac{\mathrm{MSE}_c}{\sigma_c^2+\varepsilon},
\qquad
\mathrm{VRMSE}=\sqrt{\mathrm{VMSE}}.
\]

若数据提供器给出官方场方差，NAVIER-CFD 优先使用该统计量；否则从目标数组估计。结果中必须记录统计量来源。

## Update Ratio

对于“模拟数据预训练 + 真实数据微调”：

\[
\mathrm{Update\ Ratio}=\frac{N_1}{N_2},
\]

其中 \(N_1\) 为达到目标 RMSE 所需的微调更新次数，\(N_2\) 为从头训练达到相同阈值所需的更新次数。

- 越低越好；
- 小于 \(1\) 表示预训练减少了训练更新次数；
- 只有在性能阈值与训练协议完全一致时才可比较。

## API

```python
from navier_cfd import MetricContext, MetricSuite

results = MetricSuite.from_name("data_standard").evaluate(
    prediction,
    target,
    context=MetricContext(mask=fluid_mask),
)
```
