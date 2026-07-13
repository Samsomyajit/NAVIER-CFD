# 数据集配置原生模型套件

NAVIER-CFD 0.5 提供 **52 个可执行原生参考模型**，覆盖算子学习、几何深度学习、Transformer、图网络、物理信息机器学习、生成式场模型、校正器、预条件器、自适应和不确定性量化。

## 将数据集作为导入参数

```python
from navier_cfd import load_model

model = load_model("fno", dataset="pdebench")
model = load_model("transolver", dataset="airfrans")
model = load_model("gino", dataset="drivaerml")
model = load_model("p3d", dataset="scalarflow")
```

数据集配置会给出默认的：

- 空间维度与坐标维度；
- 结构化、点云或非结构网格表示；
- 输入与输出通道；
- Fourier 模态、图邻居数或注意力参数；
- 隐藏宽度和网络层数；
- 场、坐标、图或 branch/trunk 前向模式；
- 张量布局和归一化说明。

实际样本形状和用户显式覆盖拥有更高优先级。

## 原生模型类别

### 神经算子与算子学习

DeepONet、MIONet、Fourier-DeepONet、Nested Fourier-DeepONet、Fourier-MIONet、FNO、PINO、Geo-FNO、GINO、U-FNO、F-FNO、U-NO、LSM、GNOT、Galerkin Transformer、MWT、FactFormer、ONO、Transolver、Laplace NO 和 State-Space NO。

### 物理信息机器学习

PINN、NSFnets、PINNsFormer、PINO、PIBERT、PI-MFM、RiemannONet 和 DeepM&Mnet。

### 几何、图、Transformer 与基础模型风格

MeshGraphNets、DoMINO 参考实现、UPT、DPOT、Poseidon、PROSE-FD、BCAT、PDEformer-1、P3D、AeroTransformer、Tadpole 和 ReViT。

### 生成、校正、预条件、自适应与不确定性

FourierFlow、PDE-Refiner、Solver-in-the-Loop、INC、NeuroSEM、神经算子预条件 Newton、几何感知预条件器、Conformalized-DeepONet、TANTE 风格自适应、Energy Transformer、FunDiff 和 PDE Flow Matching。

## 三个专业外部集成

PICT、diffSPH 和 NeuralDEM 仍保持为专业外部集成，因为它们需要专用的可微 CFD 或粒子求解器运行时。NAVIER-CFD 不使用普通场网络冒充这些数值求解器。

## 原生参考实现的含义

原生参考模型满足：

- 可从 PyPI 包导入；
- 可在 PyTorch 中构建；
- 可使用统一数据集配置；
- 可通过 `CFDTrainer` 训练；
- 可保存与恢复检查点；
- 通过前向和反向传播测试；
- 可运行适配器一致性检查。

它不自动表示原作者代码的逐位复现。论文级复现仍需核对官方仓库修订版、数据划分、预处理、损失函数、权重、随机种子和评估协议。

## 一致性测试

```python
from navier_cfd import validate_model_adapter

report = validate_model_adapter(
    "transolver",
    sample,
    dataset="airfrans",
)

assert report.passed
```

测试内容包括注册状态、依赖、构造、参数量、前向传播、输出形状和反向传播。
