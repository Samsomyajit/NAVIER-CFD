# 模型目录

NAVIER-CFD 使用统一模型卡描述神经 PDE/CFD 方法。每个条目记录模型标识、类别、架构、任务、物理范围、网格类型、几何模式、时间模式、维度、优势、限制、引用、代码仓库、内存需求和运行状态。

## 目录用途

模型目录用于：

- 搜索和筛选候选模型；
- 建立数据集—模型兼容性关系；
- 为推荐器提供架构先验；
- 记录原生实现和外部适配器状态；
- 连接论文证据与基准结果；
- 生成可审计的实验计划。

## 主要类别

| 类别 | 代表模型 |
|---|---|
| 物理信息学习 | PINN、NSFnets、PINNsFormer、PINO、PIBERT |
| 神经算子 | DeepONet、FNO、GINO、Geo-FNO、U-NO、LSM |
| Transformer 与图模型 | Transolver、GNOT、UPT、MeshGraphNets |
| 基础模型风格 | DPOT、Poseidon、PROSE-FD、BCAT、PDEformer-1 |
| CFD 专用模型 | PIBERT、P3D、AeroTransformer、DoMINO |
| 求解器加速 | Solver-in-the-Loop、INC、NeuroSEM、神经预条件器 |
| 生成与不确定性 | FourierFlow、PDE-Refiner、FunDiff、Conformalized-DeepONet |
| 粒子与多相 | diffSPH、NeuralDEM、Fourier-MIONet 系列 |

## Python 使用

```python
from navier_cfd import Catalog

catalog = Catalog.load_builtin()

for model in catalog.models:
    print(model.id, model.name, model.categories)
```

按类别筛选：

```python
operators = catalog.models_by_category("general_pde_solver")
```

文本搜索：

```python
results = catalog.search_models("geometry")
```

## 运行状态与科学证据

目录中的 `integration` 字段描述软件可执行状态，而不是论文性能等级。性能证据由独立的论文证据层记录，并包含数据集、指标、基线、实验条件、代码可用性和证据质量。

!!! warning "不要将目录数量等同于论文复现数量"
    模型条目、可执行参考实现、官方作者代码复现和论文数值复现是不同层次。NAVIER-CFD 在界面和文档中分别报告这些状态。
