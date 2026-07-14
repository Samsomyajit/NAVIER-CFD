# 通用数据集提供器

NAVIER-CFD 0.8.0 为内置目录中的每个数据集提供可执行加载路径，并统一返回标准 `CFDSample`。系统根据真实的上游存储、分发和许可方式选择提供器，不再假设所有数据都能通过普通 Hugging Face 表格接口读取。

## 提供器矩阵

| 数据集 | 运行时提供器 | 访问方式 | 小算力控制 |
|---|---|---|---|
| PDEBench | 科学 HDF5 选择性提供器 | 从指定 PDEBench 仓库下载一个匹配的 HDF5 文件 | 文件名/模式、大小限制、轨迹与窗口限制 |
| CFDBench | 案例压缩包提供器 | 从 `chen-yingfa/CFDBench-raw` 下载一个 ZIP 并安全解压 | 场景、案例、大小限制、步长与样本限制 |
| RealPDEBench | Arrow 分片提供器 | 下载选定 Arrow 分片并解码轨迹字节字段 | 场景/类型、分片数、大小、轨迹与窗口限制 |
| The Well | 官方 `WellDataset` | 使用官方 API 与存储选项 | 独立数据集配置、官方划分、本地缓存与流式读取 |
| APEBench | 官方程序生成提供器 | 调用已安装的 APEBench 场景 API 生成确定性轨迹 | 场景组、场景参数、样本与窗口限制 |
| AirfRANS | 官方本地 VTK 提供器 | 读取 `*_internal.vtu` 与可选的 `*_aerofoil.vtp` | 本地子集、划分清单、步长与样本限制 |
| DrivAerNet++ | 授权本地多模态提供器 | 读取本地 VTK、HDF5、NPZ/NPY、文本或安全张量导出 | 文件模式、划分清单、步长与样本限制 |
| DrivAerML | 官方本地网格提供器 | 读取 VTK/OpenFOAM 或导出的科学数组 | 文件模式、划分清单、时间范围与窗口限制 |
| ScalarFlow | 官方本地体数据提供器 | 读取 NPZ/NPY/HDF5 或导出的张量数组 | 历史长度、预测范围、步长与窗口限制 |
| ShapeNet-Car | 授权本地几何提供器 | 读取本地几何及配套 CFD 标签 | 文件模式、划分清单与样本限制 |
| EAGLE | 官方本地混合提供器 | 安全读取本地网格、数组或张量导出 | 历史长度、预测范围、步长与窗口限制 |

## 为什么部分数据集必须本地加载

AirfRANS、DrivAerNet++、DrivAerML、ScalarFlow、ShapeNet-Car 和 EAGLE 的正式数据可能需要注册、遵循许可证、手工下载，或具有极大的数据规模和特殊目录结构。因此 NAVIER-CFD 将责任分开：

1. 用户根据上游条款取得经过授权的官方数据或子集。
2. NAVIER-CFD 检查本地导出，并把支持的文件转换成训练器、模型与指标系统共用的 `CFDSample`。

“通用支持”表示每个目录数据集都有运行时适配路径，并不表示绕过注册、许可、存储或访问控制。

## 安装可选读取器

```bash
pip install "navier-cfd[scientific-data,mesh-data,torch]"
```

使用 APEBench 程序生成数据：

```bash
pip install "navier-cfd[apebench]"
```

## APEBench 程序生成

```python
from navier_cfd import load_cfd_dataset

advection = load_cfd_dataset(
    "apebench",
    configuration="Advection",
    scenario_group="difficulty",
    split="train",
    scenario_kwargs={"num_points": 128},
    n_steps_input=4,
    n_steps_output=1,
    max_samples=8,
    max_windows=64,
)
```

访问计划和样本元数据会记录 APEBench 版本、场景类、场景名称、生成划分与子集规则。

## AirfRANS 本地 VTK

```python
train = load_cfd_dataset(
    "airfrans",
    local_path="/data/AirfRANS/Dataset",
    split="train",
    max_samples=32,
)
```

提供器查找官方 `*_internal.vtu`，并在存在时合并同名 `*_aerofoil.vtp` 表面文件。输入可包含自由来流、符号距离和表面法向；目标可包含速度、压力与湍流黏度。

## 汽车点云子集

```python
cars = load_cfd_dataset(
    "drivaernetpp",
    local_path="/data/drivaernetpp_subset",
    split="train",
    file_pattern="**/*.npz",
    target_fields=("pressure", "wall_shear_stress"),
    max_samples=64,
)
```

同一本地提供器也可读取 DrivAerML 的 VTK/OpenFOAM 导出。若目录中存在 `train.txt`、`validation.txt`/`val.txt` 和 `test.txt`，系统优先使用这些清单；否则创建确定性本地子集划分，并标记为非官方划分。

## ScalarFlow 与 EAGLE 时间窗口

```python
scalar = load_cfd_dataset(
    "scalarflow",
    local_path="/data/scalarflow_subset",
    split="train",
    target_fields=("density", "velocity"),
    n_steps_input=4,
    n_steps_output=4,
    window_stride=4,
    max_samples=4,
    max_windows=128,
)
```

时间优先数组会被转换为通道最后、时间通道展平的输入与目标张量，与现有 NAVIER-CFD 结构化数据契约一致。

## 加载前检查

```python
from navier_cfd import LocalScientificDatasetManager

manager = LocalScientificDatasetManager()
status = manager.probe(
    "eagle",
    local_path="/data/eagle_subset",
    file_pattern="**/*.pt",
)
print(status.to_dict())
```

检查结果会报告路径、识别文件数量、格式统计和可选依赖，不需要打开完整数据集。

## 格式与安全

本地提供器支持 NPZ、NPY、CSV、TXT、DAT、HDF5、VTK、VTP、VTU、PLY、STL、OpenFOAM 标记文件与 PyTorch 张量导出。

安全规则：

- NumPy 使用 `allow_pickle=False`。
- PyTorch 文件要求 `weights_only=True`。
- 不执行任意 pickle 内容。
- CFDBench ZIP 解压会验证所有目标路径。
- 访问计划记录文件与划分来源，但不记录凭证。

## 验证范围

CI 使用确定性程序轨迹、结构化时间序列和点云夹具，不下载完整上游集合。因此验证分为两层：

- **契约验证：** CI 检查路由、解析、标准形状、划分来源和安全行为。
- **上游验证：** 仍应在目标机器上对真实子集执行 probe 与 smoke test，因为上游文件、许可和目录布局可能独立变化。
