# 数据集目录与提供器访问

NAVIER-CFD 将 11 个 PDE/CFD 数据集家族登记为一等对象。数据集卡记录物理问题、维度、表示方式、几何与时间模式、数据提供器、划分策略和访问契约。

| 数据集 | 主要任务 | 提供器与访问策略 |
|---|---|---|
| PDEBench | 广泛的时变 PDE 基准 | 选择性下载 Hugging Face HDF5 文件，并使用 NAVIER-CFD 轨迹适配器 |
| CFDBench | 边界、物性与几何变化 | 从 `chen-yingfa/CFDBench-raw` 选择性下载案例 ZIP，并安全解析 |
| RealPDEBench | 配对仿真与真实测量 | 从 `AI4Science-WestlakeU/RealPDEBench` 选择性下载场景 Arrow 分片 |
| The Well | 跨物理仿真数据与预训练 | 官方 `the_well.data.WellDataset`；基础路径 `hf://datasets/polymathic-ai/` 加具体 `well_dataset_name` |
| AirfRANS | 二维 RANS 与几何泛化 | 上游来源与 NAVIER-CFD 适配器 |
| APEBench | 自回归可微 PDE 基准 | 上游生成器与适配器 |
| DrivAerNet++ / DrivAerML | 真实三维车辆空气动力学 | 上游来源与几何适配器 |
| ScalarFlow | 真实体积标量输运 | 上游来源与结构化适配器 |
| ShapeNet-Car | 几何设计 | 上游来源与点云适配器 |
| EAGLE | 流体预测与重建 | 上游来源与混合适配器 |

## 为什么需要提供器感知访问

Hugging Face 仓库可能包含 Parquet、JSON、科学 HDF5、已保存的 Arrow 数据集、ZIP 压缩包或官方提供器布局。仓库可以访问并不代表可以直接调用 `datasets.load_dataset(repo_id)`。NAVIER-CFD 会先检测存储布局，再把科学格式路由到相应读取器。

```bash
navier datasets auth-status
navier datasets probe pdebench --configuration burgers
navier datasets probe cfdbench --configuration cavity
navier datasets probe realpdebench --configuration cylinder
```

鉴权优先级：

```text
显式 token 参数
    ↓
HF_TOKEN 环境变量
    ↓
`hf auth login` 保存的凭证
    ↓
匿名访问
```

Token 不会写入数据访问计划、实验清单、日志、检查点或文档输出。

## PDEBench 小子集

```python
from navier_cfd import load_cfd_dataset

burgers = load_cfd_dataset(
    "pdebench",
    configuration="burgers",
    split="train",
    file_pattern="*.h5",
    trajectory_limit=32,
    max_windows=128,
    n_steps_input=4,
    n_steps_output=1,
)
```

PDEBench 使用科学 HDF5 文件。NAVIER-CFD 选择性下载一个文件，以惰性方式读取，建立时间窗口，并返回标准 `CFDSample`。正式实验应指定准确的 `filename` 或 `file_pattern`，并固定 `revision`。

## CFDBench 小案例

```python
cavity = load_cfd_dataset(
    "cfdbench",
    configuration="cavity",
    split="train",
    case=2,
    max_samples=64,
    temporal_pairs=True,
)
```

加载器只下载一个场景案例 ZIP，验证解压路径，并适配 NPZ、NPY、CSV、TXT 或 DAT 字段。系统不会执行 pickle 内容。小案例划分是在所选压缩包内部进行的确定性划分，不应自动等同于论文中的官方案例级划分。

## RealPDEBench 小子集

```python
cylinder = load_cfd_dataset(
    "realpdebench",
    configuration="cylinder",
    data_type="real",
    split="train",
    max_arrow_files=1,
    trajectory_limit=16,
    max_windows=64,
    n_steps_input=20,
    n_steps_output=20,
)
```

小算力模式选择性下载 Arrow 分片并建立轨迹窗口。由于只加载场景的一部分，其训练、验证和测试划分会明确标记为“子集划分”，而不是完整基准的官方划分。

## The Well 官方提供器

```python
active_matter = load_cfd_dataset(
    "the_well",
    configuration="active_matter",
    split="train",
    streaming=True,
    n_steps_input=4,
    n_steps_output=1,
)
```

The Well 继续使用官方提供器。需要鉴权时，解析后的凭证仅通过存储选项传递，不会被序列化。

## 可复现性规则

- 报告结果时必须固定 Hub 修订版本；
- 记录仓库、文件或压缩包、配置名称和解析后的修订 SHA；
- 提供器存在官方划分时应保留官方划分；
- 部分文件实验必须标记为子集基准；
- 不能把同一轨迹中的重叠窗口放入训练集和测试集；
- 实验清单必须记录归一化、单位、掩码、边界、历史长度、预测范围和划分策略。
