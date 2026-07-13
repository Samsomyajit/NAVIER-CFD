# AirfRANS 几何泛化案例

本案例比较模型在未见翼型几何和运行条件上的稳态 RANS 场预测。

## 推荐模型

Transolver、GINO、Geo-FNO、MeshGraphNets、GNOT 和 DoMINO 参考实现。

```python
model = load_model("transolver", dataset="airfrans", sample=sample)
```

## 严格划分

- 按翼型几何分组；
- 按攻角或雷诺数留出；
- 保证同一几何的网格变体不会跨训练和测试；
- 使用真实点数和有效点掩码。

## 指标

- 速度、压力和湍流变量场误差；
- 翼型表面压力分布；
- 阻力和升力系数；
- 几何外推距离与误差关系；
- 推理时间和显存。

应单独分析前缘、尾缘、边界层和分离区域，因为全局平均误差可能掩盖最重要的局部失败。
