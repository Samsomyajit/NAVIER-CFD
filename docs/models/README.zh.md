# 模型图谱

本页按科学角色组织 NAVIER-CFD 模型，而不是仅按论文名称排列。

## 1. 数据驱动代理模型

目标是学习从输入条件、初始场或几何到 CFD 解场的映射。代表方法包括 FNO、DeepONet、GINO、Transolver、UPT、P3D 和 AeroTransformer。

适合：

- 参数化流场预测；
- 几何变化下的稳态 RANS；
- 时间序列预测；
- 快速设计筛选。

## 2. 物理信息模型

PINN、NSFnets、PINO、PIBERT、PI-MFM 和 RiemannONet 将 PDE、边界条件、守恒或物理偏置加入训练或注意力机制。

适合：

- 数据稀缺问题；
- 反问题；
- 物理约束代理；
- 数据—方程联合训练。

## 3. 几何和非结构表示

Geo-FNO、GINO、MeshGraphNets、Transolver、GNOT 和 DoMINO 面向点云、网格节点或变化几何。

关键检查：

- 几何编码是否与数据集一致；
- 是否支持跨网格或跨几何泛化；
- 边界节点和表面法向是否显式输入；
- 点数变化时是否使用掩码和批处理填充。

## 4. 求解器加速与校正

Solver-in-the-Loop、INC、NeuroSEM、神经预条件 Newton 和几何感知预条件器不是简单的全场代理，而是嵌入数值求解流程。

评价时应同时报告：

- 误差；
- 收敛迭代数；
- 稳定性；
- 端到端运行时间；
- 训练和部署成本；
- 盈亏平衡点。

## 5. 生成模型和不确定性

FourierFlow、PDE-Refiner、FunDiff、Flow Matching for PDEs 和 Conformalized-DeepONet 用于随机预测、后验分布、细化或区间估计。

## 选择模型

推荐先定义任务：

```python
from navier_cfd import TaskSpec, recommend_models

# 建立 TaskSpec 后使用证据感知推荐器
```

然后用 `dataset=` 构造候选模型，并在同一正式数据划分和指标协议下比较。
