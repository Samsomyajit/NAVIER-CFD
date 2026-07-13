# 可执行模型中心

`ModelHub` 为 NAVIER-CFD 中的模型提供统一的发现、状态检查、构造和外部实现适配接口。

## 安装

```bash
pip install navier-cfd
```

需要运行原生 PyTorch 模型时：

```bash
pip install "navier-cfd[models]"
```

## 查看模型状态

```python
from navier_cfd import list_models

for handle in list_models():
    status = handle.status
    print(handle.id, status.mode, status.executable, status.dependency_available)
```

运行状态包括：

| 状态 | 含义 |
|---|---|
| `native` | NAVIER-CFD 内置可执行参考实现 |
| `external_adapter` | 已注册稳定的外部 Python 构造入口 |
| `source_available` | 已知官方源码，但尚无稳定构造入口 |
| `metadata` | 只有模型卡和证据信息 |

## 数据集感知构造

```python
from navier_cfd import load_model

fno = load_model("fno", dataset="cfdbench")
transolver = load_model("transolver", dataset="airfrans")
```

返回的模型包含：

```python
model.navier_model_id
model.navier_dataset_id
model.navier_build_plan
model.navier_dataset_configuration
```

## 注册项目内构造器

```python
from navier_cfd import ModelHub

hub = ModelHub()

def build_my_gino(*, task, spec, width=128, **kwargs):
    return MyGINO(dimension=task.dimension, width=width, **kwargs)

hub.register_builder("gino", build_my_gino)
model = hub.load("gino", task=task, width=256)
```

## 连接外部实现

```python
hub.register_external(
    "transolver",
    entrypoint="my_transolver_package:Transolver",
    install_spec="my-transolver-package",
)

model = hub.load("transolver", hidden_dim=256, num_layers=8)
```

NAVIER-CFD 不会在导入包时自动克隆或执行第三方仓库。安装第三方依赖必须显式允许：

```python
hub.model("transolver").install(allow_external=True)
```

## 科学注意事项

统一加载器只标准化构造和状态管理，不能消除不同模型的科学差异。每个实验仍需明确：

- 输入和输出变量；
- 无量纲化与归一化；
- 网格、点云或粒子表示；
- 边界条件编码；
- 时间推进方式；
- 损失函数和训练协议；
- 官方权重、许可证和引用要求。
