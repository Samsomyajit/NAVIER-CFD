# 数据集目录

NAVIER-CFD 将 11 个 PDE/CFD 数据集登记为一等对象。数据集卡描述物理问题、维度、网格类型、几何变化、时间模式、下载位置和适用模型类别。

## 已注册数据集

| 数据集 | 表示 | 默认维度 | 主要任务 |
|---|---|---:|---|
| PDEBench | 结构化 | 2D，实际样本支持 1D–3D | PDE 预测与算子学习 |
| CFDBench | 结构化 | 2D | 方腔、管流、溃坝、圆柱绕流 |
| RealPDEBench | 结构化 | 2D | 仿真到真实系统预测 |
| The Well | 结构化 | 3D 默认 | 大规模多物理场 |
| APEBench | 结构化 | 2D 默认 | 自回归 PDE 模拟器 |
| ScalarFlow | 结构化体数据 | 3D | 标量输运 |
| AirfRANS | 点云/非结构 | 2D | 翼型 RANS 与几何泛化 |
| DrivAerNet++ | 点云 | 3D | 车辆空气动力学 |
| DrivAerML | 非结构网格 | 3D | 高保真车辆 CFD |
| ShapeNet-Car | 点云 | 3D | 几何条件车辆场预测 |
| EAGLE | 结构化/非结构 | 2D–3D | 流体与几何学习 |

## 使用内置配置

```python
from navier_cfd import configure_model_for_dataset

plan = configure_model_for_dataset(
    "transolver",
    "airfrans",
)

print(plan.dataset_configuration)
print(plan.builder_kwargs)
```

## 使用真实样本

数据集版本可能在变量命名、通道顺序、分辨率和目标定义上不同，因此正式实验应传入真实样本：

```python
model, plan = load_model(
    "fno",
    dataset="pdebench",
    sample=sample,
    return_plan=True,
)
```

## 数据划分原则

不要在以下场景中仅使用随机逐样本划分：

- 同一几何的相邻时间帧；
- 同一雷诺数附近的连续工况；
- 同一车辆或翼型的网格变体；
- 仿真和真实数据成对样本。

应优先采用按几何、工况、时间窗口、实验批次或物理参数分组的严格留出测试。

## 归一化与单位

每个实验应保存：

- 原始物理单位；
- 无量纲化定义；
- 训练集统计量；
- 掩码和边界定义；
- 输入历史长度与预测范围；
- 数据集版本和文件校验信息。
