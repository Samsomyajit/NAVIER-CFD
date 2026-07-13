# 模型推荐系统

NAVIER-CFD 推荐器根据具体 CFD/PDE 任务筛选不兼容模型，并结合架构匹配与论文级证据进行排序。

## 定义任务

```python
from navier_cfd import Catalog, TaskSpec, recommend_models

catalog = Catalog.load_builtin()
task = TaskSpec(
    problem="3d_vehicle_aerodynamics",
    task_type="surrogate",
    dimension=3,
    mesh_type="point_cloud",
    temporal_mode="steady",
    geometry_mode="varying",
    physics=("aerodynamics",),
    fidelity="rans",
    requires_geometry_transfer=True,
    requires_mesh_transfer=True,
    hardware_memory_gb=80,
)

results = recommend_models(task, catalog.models, top_k=8)
```

## 两阶段逻辑

### 1. 硬兼容性筛选

首先检查：

- 维度；
- 网格或点云表示；
- 固定或变化几何；
- 稳态、非稳态或自回归时间模式；
- 代理、校正器、预条件器等数值角色；
- 显存限制；
- 守恒、不确定性和迁移要求。

明显不适合的模型不会仅因论文分数高而进入排名。

### 2. 兼容候选排序

排序综合：

- 架构兼容性；
- 任务与论文实验的相似度；
- 指标方向和基线改进；
- 证据质量；
- 证据数量与覆盖度；
- 贝叶斯收缩后的置信度。

## 输出解释

```python
for result in results:
    print(result.model.name)
    print("final:", result.score)
    print("architecture:", result.architecture_score)
    print("evidence:", result.evidence_score)
    print("confidence:", result.evidence_confidence)
    print("coverage:", result.evidence_coverage)
    print("reasons:", result.reasons)
    print("cautions:", result.cautions)
```

推荐结果是实验优先级建议，不是普适排名。最终选择仍应在目标数据集、正式划分和相同计算预算下验证。

## 命令行

```bash
navier recommend \
  --problem vehicle_drag \
  --task surrogate \
  --dimension 3 \
  --mesh point_cloud \
  --temporal steady \
  --geometry varying \
  --physics aerodynamics \
  --fidelity rans \
  --memory-gb 80
```

浏览器推荐器运行在客户端，并能够导出可复现任务清单。
