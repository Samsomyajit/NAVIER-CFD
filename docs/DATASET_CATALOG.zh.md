# 数据集目录与提供器访问

NAVIER-CFD 将 11 个 PDE/CFD 数据集家族登记为一等对象。数据集卡记录物理问题、维度、表示方式、几何与时间模式、数据提供器、官方划分和访问契约。

| 数据集 | 主要任务 | 提供器与访问方式 |
|---|---|---|
| PDEBench | 广泛的时变 PDE 基准 | Hugging Face：`AI4Science-WestlakeU/PDEBench` |
| CFDBench | 边界、物性与几何变化 | Hugging Face：`chen-yingfa/CFDBench` |
| RealPDEBench | 配对仿真与真实测量 | Hugging Face：`AI4Science-WestlakeU/RealPDEBench` |
| The Well | 跨物理仿真数据与预训练 | 官方 `the_well.data.WellDataset`；基础路径 `hf://datasets/polymathic-ai/` 加具体 `well_dataset_name` |
| AirfRANS | 二维 RANS 与几何泛化 | 上游来源与 NAVIER-CFD 适配器 |
| APEBench | 自回归可微 PDE 基准 | 上游生成器与适配器 |
| DrivAerNet++ / DrivAerML | 真实三维车辆空气动力学 | 上游来源与几何适配器 |
| ScalarFlow | 真实体积标量输运 | 上游来源与结构化适配器 |
| ShapeNet-Car | 几何设计 | 上游来源与点云适配器 |
| EAGLE | 流体预测与重建 | 上游来源与混合适配器 |

## 通用 Hugging Face 访问

```python
from navier_cfd.datasets import HuggingFaceDatasetManager

manager = HuggingFaceDatasetManager(token=None)
print(manager.discover("CFD fluid dynamics", limit=50))
manager.download("chen-yingfa/CFDBench", "./data/cfdbench", revision="<commit>")
stream = manager.load(
    "AI4Science-WestlakeU/RealPDEBench",
    split="train",
    streaming=True,
)
```

## The Well 原生提供器访问

The Well 不是一个单独的 `datasets.load_dataset` 仓库。应通过提供器分发接口选择具体配置：

```python
from navier_cfd import load_cfd_dataset

active_matter = load_cfd_dataset(
    "the_well",
    configuration="active_matter",
    split="train",
    streaming=True,
    n_steps_input=4,
    n_steps_output=1,
)
```

命令行下载：

```bash
navier datasets well-list
navier datasets download the_well \
  --configuration active_matter \
  --split train \
  --local-dir ./data/the_well
```

## 数据划分原则

存在官方划分时必须优先保留。不能把同一轨迹中的重叠时间窗口随机分配到训练集和测试集。几何泛化应按几何身份划分；工况泛化应按案例或参数区间划分。

实验清单应记录提供器版本、数据集修订、配置名称、归一化统计量、单位、掩码、边界、历史长度、预测范围和划分策略。