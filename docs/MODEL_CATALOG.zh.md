# 模型目录

NAVIER-CFD 使用统一模型卡描述神经 PDE/CFD 方法。内置目录包含 55 个模型，并采用多标签分类。下表为各模型家族中列出的代表模型补充了主要原始文献。文献用于说明科学方法的来源；并不表示每个 NAVIER-CFD 原生参考实现都与作者代码、私有预处理或检查点完全一致。

| 模型家族 | NAVIER-CFD 中的代表模型 | 代表性原始文献 |
|---|---|---|
| 物理信息神经场 | PINN、NSFnets、PINNsFormer、PINO、PIBERT | Raissi、Perdikaris 与 Karniadakis，*JCP* 2019，[Physics-informed neural networks](https://doi.org/10.1016/j.jcp.2018.10.045)；Jin 等，*JCP* 2021，[NSFnets](https://doi.org/10.1016/j.jcp.2020.109951)；Zhao、Ding 与 Prakash，[PINNsFormer](https://arxiv.org/abs/2307.11833)；Li 等，[PINO](https://arxiv.org/abs/2111.03794)；Chakraborty、Pan 与 Chen，PIBERT（2026），[项目源码](https://github.com/Samsomyajit/pibert)。 |
| 分支—主干与谱神经算子 | DeepONet、MIONet、Fourier-DeepONet、Fourier-MIONet、FNO、F-FNO、U-FNO、U-NO、LSM | Lu 等，*Nature Machine Intelligence* 2021，[DeepONet](https://doi.org/10.1038/s42256-021-00302-5)；Jin、Meng 与 Lu，[MIONet](https://arxiv.org/abs/2202.06137)；Li 等，[FNO](https://arxiv.org/abs/2010.08895)；Tran 等，[F-FNO](https://arxiv.org/abs/2111.13802)；Rahman、Ross 与 Azizzadenesheli，[U-NO](https://arxiv.org/abs/2204.11127)。 |
| 几何、图与网格算子 | Geo-FNO、GINO、MeshGraphNets、Transolver、GNOT、UPT、DoMINO、ReViT | Li 等，[Geo-FNO](https://arxiv.org/abs/2207.05209)；Li 等，[GINO](https://arxiv.org/abs/2309.00583)；Pfaff 等，[MeshGraphNets](https://arxiv.org/abs/2010.03409)；Wu 等，[Transolver](https://arxiv.org/abs/2402.02366)。使用 GNOT、UPT、DoMINO 或 ReViT 时，应进一步引用实验所对应的原始论文或官方项目。 |
| Transformer 与基础模型风格 PDE 模型 | DPOT、Poseidon、PROSE-FD、BCAT、PDEformer-1、PI-MFM、P3D、AeroTransformer、Tadpole | Liu 等，[PROSE-FD](https://arxiv.org/abs/2409.09811)；Liu、Sun 与 Schaeffer，[BCAT](https://arxiv.org/abs/2501.18972)；Ye 等，[PDEformer-1](https://arxiv.org/abs/2407.06664)。对于 DPOT、Poseidon、PI-MFM、P3D、AeroTransformer 和 Tadpole，应引用实验中实际采用版本的原始论文。 |
| 生成式与概率 PDE 模型 | FourierFlow、PDE-Refiner、FunDiff、Flow Matching for PDEs、Conformalized-DeepONet | Wang 等，[FourierFlow](https://arxiv.org/abs/2506.00862)；Lippe 等，[PDE-Refiner](https://arxiv.org/abs/2308.05732)。使用 FunDiff、流匹配或保形算子时，应引用相应的原始论文。 |
| 混合数值加速 | Solver-in-the-Loop、INC、PICT、NeuroSEM、神经算子预条件 Newton、几何感知神经预条件器 | Um 等，[Solver-in-the-Loop](https://arxiv.org/abs/2007.00016)；Wei 等，INC（NeurIPS 2025），[官方实现](https://github.com/tum-pbs/INC)；Franz 等，PICT，*Journal of Computational Physics* 2025。使用 NeuroSEM 或神经预条件器时，应引用对应的原始论文。 |
| 粒子与多相学习 | diffSPH、NeuralDEM、Nested Fourier-DeepONet、Fourier-MIONet、U-FNO | Winchenbach 与 Thuerey，diffSPH，*Journal of Computational Physics* 2026；Wen 等，U-FNO，*Advances in Water Resources* 2022。使用 NeuralDEM 或 Fourier 算子变体时，应引用所选实现和数据集对应的官方来源。 |
| 专用耦合与逆问题模型 | DeepM&Mnet、RiemannONet、Energy Transformer 流场重建、TANTE | Mao 等，*JCP* 2021，[DeepM&Mnet](https://doi.org/10.1016/j.jcp.2021.110698)；Wu 等，TANTE，*JCP* 562（2026），115041。报告 RiemannONet 或 Energy Transformer 结果时，应引用其原始论文。 |

## 引用规范

使用 NAVIER-CFD 发表研究结果时，建议同时引用：

1. **NAVIER-CFD**：用于模型集成、数据集配置、训练、评估或推荐流程；
2. **原始模型论文**：用于所采用的每一种架构；
3. **原始数据集或基准论文**；
4. **官方上游实现和检查点**（如适用）。

模型卡、NAVIER-CFD 原生参考实现、作者官方实现和论文数值复现属于不同的证据层级。模型注册表和推荐器显示的 `reference` 字段用于保留原始归属。

## Python 使用

```python
from navier_cfd import Catalog

catalog = Catalog.load_builtin()

for model in catalog.models:
    print(model.id, model.name, model.categories, model.reference)
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