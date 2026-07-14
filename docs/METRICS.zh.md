# 评价指标

NAVIER-CFD 0.6 增加了用于数据精度、谱保真度、物理一致性和训练效率的命名指标套件。每个指标结果同时记录优化方向、理想值、假设条件、有效性和评价空间。

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

## 内置指标套件

| 套件 | 用途 | 主要指标 |
|---|---|---|
| `data_standard` | 通用场预测精度 | MSE、RMSE、MAE、L∞、相对 L1/L2、NMSE、NRMSE、R²、Pearson、余弦相似度 |
| `the_well` | 与 The Well 对齐的归一化和谱评价 | L∞、MAE、MSE、RMSE、NMSE、NRMSE、Pearson、VMSE、VRMSE、分频段谱 MSE |
| `realpdebench` | RealPDEBench 风格的数据和物理评价 | RMSE、MAE、逐样本相对 L2、R²、fRMSE、FE、湍动能、MVPE、Update Ratio |
| `fluid_standard` | CFD 物理一致性 | RMSE、相对 L2、谱相对误差、散度误差、动能误差、涡量 RMSE |

## 指标结果结构

每个指标返回 `MetricResult`：

```python
{
    "name": "kinetic_energy_relative_error",
    "value": 0.031,
    "category": "physics",
    "direction": "lower",
    "best_value": 0.0,
    "valid": True,
    "assumptions": ["已声明速度通道"],
    "metadata": {"evaluation_space": "physical"},
}
```

若缺少必需元数据，指标会被标记为无效，而不是生成没有物理意义的数值。例如，计算阻力必须具备表面法向量、面积、压力/剪切场和参考量。

## 评价空间

在明确说明时，数据指标可以在归一化空间中计算。物理指标通常应在反归一化后的物理单位中计算。实验清单应记录归一化方法、统计量来源、掩码策略、坐标轴、通道映射和聚合方式。

## 来源与兼容性

该指标包吸收了 The Well 和 RealPDEBench 的优秀设计，但保留明确的兼容模式。指标名称相同并不意味着归一化、FFT 约定、频段划分、探针位置或聚合方式完全相同。复现实验必须记录这些细节并引用原始基准。

- The Well API：<https://polymathic-ai.org/the_well/api/>
- RealPDEBench 数据指标：<https://realpdebench.github.io/metrics/data-oriented/>
- RealPDEBench 物理指标：<https://realpdebench.github.io/metrics/physics-oriented/>
