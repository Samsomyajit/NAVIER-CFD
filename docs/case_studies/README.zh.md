# 案例研究

案例研究展示如何将 NAVIER-CFD 的数据集适配、模型配置、训练、指标和证据层应用到具体 CFD 问题。

## 可用案例

- **PDEBench Navier–Stokes**：结构化时序场、长时间滚动和谱误差。
- **CFDBench 方腔流**：边界条件和物性变化下的代理建模。
- **CFDBench 圆柱绕流**：涡脱落、压力和速度预测。
- **RealPDEBench 圆柱案例**：仿真到真实迁移。
- **RealPDEBench 流固耦合**：流体和结构响应联合预测。
- **AirfRANS 几何泛化**：未见翼型与运行条件。
- **汽车三维空气动力学**：点云和非结构几何场预测。
- **混合加速**：神经校正器、预条件器和求解器耦合。

## 统一案例模板

每个案例应明确：

1. 科学问题和假设；
2. 数据集版本与字段；
3. 严格数据划分；
4. 候选模型和选择理由；
5. 训练预算与随机种子；
6. 场、谱、守恒和积分量指标；
7. 运行时间、显存和盈亏平衡；
8. 失败案例和限制；
9. 检查点、配置和结果清单。

## 示例

```python
from navier_cfd import Experiment, TaskSpec, TrainerConfig

experiment = Experiment(
    dataset_id="cfdbench",
    model_id="pino",
    task=TaskSpec(
        problem="cylinder",
        task_type="forecasting",
        dimension=2,
        mesh_type="structured",
        temporal_mode="autoregressive",
        geometry_mode="fixed",
        physics=("incompressible_navier_stokes",),
    ),
    trainer_config=TrainerConfig(epochs=100, optimizer="adamw"),
)
```

未提供独立中文译文的个别详细案例页面会自动回退到英文原文，页面顶部的语言选择器仍可用于返回中文页面。
