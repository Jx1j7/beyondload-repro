# CPU 指标复算：独立审计结果

原始 `experiments/results.csv` 未修改。本报告使用其逐次运行的已保存 MAE；没有重新训练或推理。

- 纳入：去重后 step=1 结果 397 行，14 个站点数据集。
- 风电和负荷分子：归档的全时段 MAE；光伏分子：归档的实际功率 >1% 时段 MAE。
- 分母：训练段内相隔 24 小时的季节朴素绝对误差均值；光伏分母限实际功率 >1% 时段。
- 训练段终点：各站序列长度的 70%；缺失配对不计入分母。
- 该复算无法验证未来 NWP 的发行时刻，也不能从汇总 MAE 恢复逐起点误差或置信区间。

## 主结果敏感性

- 风光站点观测最优：{'lightgbm+cov': 13}。
- Friedman（5 模型 × 14 数据集）：χ²=35.26，名义 p=4.11e-07。
- 排除不同模型有效起点不一致的 wind zone 3 后，Friedman（5 模型 × 13 数据集）：χ²=32.00，名义 p=1.91e-06。
- 完整覆盖模型平均秩：lightgbm+cov 1.00; chronos2 2.86; patchtst+cov 3.07; chronos 3.79; itransformer+cov 4.29

## 跨站均值（标准化分母口径）

| 轨道 | 模型 | 站点数 | 平均 MASE |
|---|---|---:|---:|
| load | lightgbm+cov | 1 | 0.378 |
| load | chronos | 1 | 0.756 |
| load | chronos2 | 1 | 0.947 |
| load | patchtst+cov | 1 | 0.960 |
| load | itransformer+cov | 1 | 1.420 |
| solar | lightgbm+cov | 3 | 0.605 |
| solar | chronos2 | 3 | 0.862 |
| solar | chronos | 3 | 0.877 |
| solar | moirai | 3 | 0.887 |
| solar | ttm | 3 | 0.953 |
| solar | itransformer+cov | 3 | 0.988 |
| solar | dlinear | 2 | 0.990 |
| solar | timesfm | 3 | 1.002 |
| solar | timemixer+cov | 2 | 1.099 |
| solar | patchtst+cov | 3 | 1.186 |
| wind | lightgbm+cov | 10 | 0.472 |
| wind | ttm | 10 | 0.815 |
| wind | moirai | 10 | 0.823 |
| wind | timesfm | 10 | 0.836 |
| wind | patchtst+cov | 10 | 0.854 |
| wind | timemixer+cov | 10 | 0.856 |
| wind | dlinear | 10 | 0.860 |
| wind | chronos2 | 10 | 0.874 |
| wind | chronos | 10 | 0.894 |
| wind | itransformer+cov | 10 | 0.912 |

## 使用边界

这是事后修正的指标口径，不是预注册原口径。`paper/manuscript-v2.md` 的表 3 已将其用作主表，并在方法中披露修正；旧口径留在原始 `results.csv`。风区同源且可能相关，Friedman p 值与 Nemenyi CD 基于独立块假设，只作名义描述，不能据此作总体显著性推断。该处理不能修复运行时间、NWP 发布时刻或 wind zone 3 的非共同起点问题。
