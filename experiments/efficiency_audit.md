# 计算效率归档口径核查（2026-11-04）

只读检查 `code/experiments/run_efficiency.py`、`cloud_efficiency.csv`、`efficiency_all.csv`，未重新加载模型。

## 已确认的问题

1. 英文稿原句把 Apple M1 数值（Chronos-Bolt 682.33 ms、Chronos-2 83.22 ms、TimesFM 1253.91 ms）称为“云硬件轨道”。完整的 `Intel Xeon 8352V (A100 box)` 8 模型子表相应为 476.69、59.93、639.46 ms。原句混用了两台硬件。
2. 该脚本的 `--device` 默认是 `cpu`。Chronos 管道明确使用 `device_map="cpu"`，TimesFM 配置 `backend="cpu"`；“A100 box”是主机标签，不能据此称这些数字为 A100 GPU 推理时延。
3. LightGBM 的 `lat_ms_b1` 不是直接测量的推理时延，脚本 `lats_proxy()` 返回训练时间除以预测起点数。把它和神经模型的预热后函数调用时延同列作 Pareto 比较不成立。
4. `cloud_efficiency.csv` 的四条早期云记录 `peak_rss_mb` 为 0.4–0.6，而合并表同类记录为 409.6–614.4，存在单位转换差异；不应直接使用原文件的云端峰值内存列作模型排序。
5. `efficiency_all.csv` 含重复的 Apple M1 行及两个不同云主机标签的片段。必须先锁定单一硬件子表与测量协议，不能简单跨行取最小值。

## 可保留的探索性读数

下表仅为归档中单一 `Intel Xeon 8352V (A100 box)` 标签的 batch-1 读数；均应称为该主机上的**CPU 实现试点**，LightGBM 一行是不同口径的代理成本，故单独标记。

| 模型 | 归档读数 (ms) | 解释 |
|---|---:|---|
| DLinear | 0.107 | 预热后模型调用 |
| TTM | 2.987 | 预热后模型调用 |
| Moirai-2 | 11.390 | 预热后预测器调用 |
| Chronos-2 | 59.930 | 预热后 `predict_df` 调用 |
| Chronos-Bolt | 476.690 | 预热后管道调用 |
| TimesFM-2.0 | 639.460 | 预热后 `forecast` 调用 |
| LightGBM | 1.270 | **训练时间 / 起点数代理量，非直接推理测量** |

本表只支持“实现与成本读数有明显差异”的探索性观察；加载、预处理、设备、批处理语义和 LightGBM 代理口径不同，不能建立可信的跨模型精度–时延 Pareto 前沿。若日后需要正式效率结论，须统一起点、设备、批量、预热、加载边界和直接测时程序后复测。
