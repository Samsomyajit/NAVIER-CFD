# PDEBench 数据集卡

PDEBench 是广泛使用的时变偏微分方程基准，包含 Navier–Stokes、对流、反应扩散等多种系统，并提供一维、二维和三维结构化数据。

## NAVIER-CFD 默认配置

- 表示：结构化网格；
- 默认维度：2D，真实样本可覆盖 1D–3D；
- 时间模式：自回归或直接多步预测；
- 适用模型：FNO、PINO、PIBERT、DeepONet、DPOT、Poseidon 和其他算子学习模型。

```python
model = load_model("pino", dataset="pdebench", sample=sample)
```

## 建议任务

- 单步预测；
- 多步滚动；
- 参数外推；
- 分辨率迁移；
- 不同 PDE 之间的预训练与微调；
- 数据损失与物理残差联合训练。

## 评价要求

至少报告：

- 相对 \(L_2\)、RMSE 和 MAE；
- 每个时间步的误差增长；
- 最终时刻误差；
- 频谱误差；
- 守恒或 PDE 残差；
- 推理时间与显存。

正式实验应固定数据版本、空间分辨率、时间步长、训练时间窗口和官方划分。
