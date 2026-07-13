# 统一实验流程与 PIBERT

NAVIER-CFD 的统一实验层将原始数据、标准化样本、数据加载器、模型配置、训练、检查点和 CFD 指标连接成可复现流程。

## 标准工作流

```text
原始数据集
   ↓
DatasetAdapter
   ↓
CFDSample / CFDBatch
   ↓
数据集感知模型配置
   ↓
ModelHub 原生模型或外部适配器
   ↓
CFDTrainer
   ↓
检查点、指标和实验清单
```

## 规范化样本

```python
from navier_cfd import CFDSample

sample = CFDSample(
    inputs=input_fields,
    targets=target_fields,
    coordinates=coordinates,
    parameters={"Re": 1000.0},
    mask=fluid_mask,
    metadata={"case": "cylinder"},
)
```

`CFDSample` 可表示结构化网格、点云和网格节点。批处理层支持可变点数、填充、有效性掩码、参数和元数据。

## 训练、验证和测试划分

```python
from navier_cfd import AdaptedDataset, make_dataloaders

dataset = AdaptedDataset(raw_dataset, adapter)
loaders = make_dataloaders(
    dataset,
    batch_size=8,
    train=0.70,
    validation=0.15,
    test=0.15,
    seed=42,
)
```

固定随机种子可保证划分可重复。对于跨几何、跨雷诺数或跨工况研究，应使用显式分组划分而不是随机泄漏相近样本。

## 高层实验 API

```python
from navier_cfd import Experiment, TaskSpec, TrainerConfig

experiment = Experiment(
    dataset_id="pdebench",
    model_id="pibert",
    task=TaskSpec(
        problem="navier_stokes",
        task_type="forecasting",
        dimension=2,
        mesh_type="structured",
        temporal_mode="autoregressive",
        geometry_mode="fixed",
        physics=("incompressible_navier_stokes",),
    ),
    trainer_config=TrainerConfig(
        epochs=100,
        optimizer="adamw",
        learning_rate=1e-3,
        mixed_precision=True,
    ),
    batch_size=8,
    output_dir="runs/pibert-pdebench",
)

result = experiment.run(raw_dataset)
print(result.metrics)
print(result.build_plan)
```

## PIBERT 原生实现

PIBERT 参考实现包括：

- Fourier 坐标嵌入；
- 多尺度小波细节特征；
- 物理偏置多头注意力；
- 坐标距离注意力偏置；
- 可选 PDE 残差偏置；
- 分块注意力以降低峰值显存；
- 结构化场和点序列输入；
- 流体域掩码和有效点掩码。

```python
from navier_cfd import load_model

model = load_model(
    "pibert",
    dataset="realpdebench",
    sample=sample,
    overrides={
        "hidden_dim": 128,
        "num_layers": 6,
        "num_heads": 8,
        "num_frequencies": 16,
    },
)
```

## 结果与检查点

检查点保存模型权重、优化器、学习率调度器、训练配置、数据集和模型标识、指标及实验元数据。生产实验应同时保存：

- 代码提交 SHA；
- 数据集修订版本和正式划分；
- 归一化统计量；
- 随机种子；
- 模型构建计划；
- 训练日志和最终测试指标。
