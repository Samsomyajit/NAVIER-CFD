# The Well 数据集卡

The Well 是一个包含多种大规模物理仿真数据集的提供器体系。NAVIER-CFD 将其作为一等数据提供器接入，同时保持自身定位不变：多数据集集成、多模型家族、数据集驱动配置、统一训练与检查点、物理指标、证据感知推荐和化工扩展。

## 正确的访问方式

The Well **不是**一个可以直接通过 `datasets.load_dataset("polymathic-ai/the_well")` 加载的单一 Hugging Face 仓库。官方接口使用 `the_well.data.WellDataset`，并要求提供基础路径和具体子数据集名称：

```python
from navier_cfd import load_cfd_dataset

dataset = load_cfd_dataset(
    "the_well",
    configuration="active_matter",
    split="train",
    streaming=True,
    n_steps_input=4,
    n_steps_output=1,
    use_normalization=True,
)
```

其对应的官方参数为：

```python
WellDataset(
    well_base_path="hf://datasets/polymathic-ai/",
    well_dataset_name="active_matter",
    well_split_name="train",
)
```

安装可选数据提供器依赖：

```bash
pip install "navier-cfd[the-well]"
```

使用本地数据时：

```python
dataset = load_cfd_dataset(
    "the_well",
    configuration="rayleigh_benard",
    split="valid",
    streaming=False,
    local_path="/data/the_well",
)
```

## 统一数据适配

The Well 官方记录中的场通常具有：

\[
[T, L_1, \ldots, L_d, F]
\]

的形状。NAVIER-CFD 默认转换为：

\[
[L_1, \ldots, L_d, TF]
\]

即把时间历史折叠到通道维，同时保留真实空间维度。适配器还会保存：

- 场名称和张量阶次；
- 输入与输出时间网格；
- 空间坐标；
- 常量标量；
- 边界条件元数据；
- 官方数据划分；
- 提供器版本和访问计划；
- 归一化来源。

```python
sample = dataset[0]
model, plan = load_model(
    "fno",
    dataset="the_well",
    sample=sample,
    return_plan=True,
)
```

实际样本会决定空间维度、输入输出通道、历史长度、坐标和模型配置。

## 官方划分与归一化

应保留 The Well 的 `train`、`valid` 和 `test` 官方划分，不能把同一轨迹中的重叠时间窗口随机分到不同集合。The Well 支持官方 Z-score 和 RMS 归一化；物理指标通常应在恢复物理单位后计算。

## 科学范围

NAVIER-CFD 将 The Well 作为高质量数据和基准后端使用，不复制其完整 15 TB 数据集，也不替代官方软件包。发表结果时，应同时引用 NAVIER-CFD、The Well 论文和具体数据集来源。

## 参考资料

- Ohana 等，*The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning*，NeurIPS 2024。
- 官方 API：<https://polymathic-ai.org/the_well/api/>
- 官方仓库：<https://github.com/PolymathicAI/the_well>
