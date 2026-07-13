# PDEBench Navier–Stokes 案例

本案例用于比较结构化时序流场上的神经算子与物理信息模型。

## 推荐设置

- 数据集：PDEBench Navier–Stokes 子集；
- 任务：给定历史场预测未来场；
- 候选模型：FNO、PINO、PIBERT、DPOT、Poseidon；
- 划分：使用官方训练、验证和测试划分；
- 训练：相同预算、多随机种子和相同归一化。

```python
model = load_model("pino", dataset="pdebench", sample=sample)
```

## 指标

- 相对 \(L_2\)、RMSE、MAE；
- 每个时间步的滚动误差；
- 最终时刻误差；
- 能量谱误差；
- 速度散度；
- 推理时间和显存。

## 消融

建议比较数据损失、物理残差、不同历史长度、不同预测步长、不同空间分辨率以及教师强制与自由滚动。

结果解释应区分短期拟合质量和长时间动力学稳定性。
