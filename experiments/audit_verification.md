# 本轮工件核验（2026-11-04）

- 英文稿表 3 的 30 个模型×轨道单元逐项对照 `metric_audit/corrected_cross_site.csv`：25 个数字的三位小数一致，5 个缺测格与 CSV 的模型覆盖范围一致；五模型统计与 `metric_audit/report.md` 一致。
- 新无 NWP LightGBM CSV 共 70 行，14 个站点数据集均有 5 个种子；加入真实 UTC 起止时间和逐行写入时间后重跑，70/70 MASE 数值完全相同（最大绝对差 0）。实际执行清单见 `information_match_cpu/run_manifest.md`，输入与输出 SHA-256 已登记。
- 归档 `results.csv` 与只读 ZCode 原目录同名文件 SHA-256 均为 `de787fa9079cab02598904398344166142d832ac734ec918e0435999c275dea8`，确认本轮未覆盖原结果。
- 新增 7 个 CPU 审计、分析、附录和制图脚本通过 `python3 -m py_compile`；网页看板经 `HTMLParser` 解析；两张正文图存在并目视检查。
- citation-verifier 本地扫描：占位符 0，正文引用编号均在 [1]–[23] 条参考文献中定义。23 条文献的存在性与主要元数据另按原始或发布机构页面作有界核查，见 `paper/reference-audit-2026-11-04.md`。
- 新 LightGBM 向量化预测与原串行实现，在风电 zone 1 和光伏 zone 1 的 `step=24` 输出逐值一致（最大绝对差 0）。这项核查不代表旧 TSFM 批量推理已经证实等价。
- 外部数据可行性检查发现旧 DKASC `.csv` 实为 HTML 页面；新版抓取脚本通过 `bash -n`，且用该 HTML 文件作本地输入时拒绝保存。Elia 脚本无显式数据集 ID 会拒绝运行；二者均不再悄悄写入 ZCode 原目录。`experiments/external_dataset_feasibility.md` 记录官方字段与使用边界。
- 对 `code/` 扫描后，旧 ECL 抓取和 SARIMA 电源哨兵脚本中的 ZCode 原目录绝对写入路径均已改为由脚本位置解析当前工作区；SARIMA 哨兵不再广域 `pkill`。两个脚本分别通过 `bash -n` / `zsh -n`。
- 最后一轮重跑指标审计后，英文稿、图 1 与审计报告均将 Friedman p 值及 Nemenyi CD 标为独立块假设下的名义结果；表 3 再核验为 25 个数值格一致、5 个缺测格一致，原 CSV 的 SHA-256 仍与 ZCode 原目录相同。

这些检查保证工件与正文数字一致及新增 CPU 轨道可追踪；它们不能恢复历史 TSFM checkpoint revision、原 NWP 发行版本、逐起点误差或冻结前每条结果的首跑时间。

## 每日共同窗口四臂对照的追加核验

- 六个新增 CPU 审计、模型运行、分析、制图和附录脚本，以及修复 `future_df` 调用的 `run_zeroshot.py`，均通过 `python -m py_compile`。
- `candidate_origins.csv` 有 13 站共 1,635 个合格起点：风电 1,164，光伏 471。逐一核对 Chronos-2 的 26 个和 LightGBM 的 130 个逐点 CSV，文件 SHA-256 均与各自运行清单一致；每个文件恰好覆盖本站全部合格起点，每个起点 24 个预测步，无重复步、非有限实际值或非有限预测值。
- 156 个逐点文件共享的 39,240 个“站点×起点×预测步”实际值与目标时间戳逐值一致；LightGBM 五种子、两个模型及两个信息臂均以相同实际目标评分。英文稿表 3c 的八个三位小数分数与 `competition_aligned_cpu/paired_cross_site.csv` 一致。
- 新模型运行的批量/串行首起点核查：LightGBM 风电和光伏两臂最大差 0；Chronos-2 两臂最大差小于 `3e-7`。这只约束新运行的样例，不追溯证明旧批处理。
- 原 `experiments/results.csv` SHA-256 仍为 `de787fa9079cab02598904398344166142d832ac734ec918e0435999c275dea8`，与 ZCode 原目录相同。

新评测保存了自己的逐起点预测和配对损失，故上文对旧归档“没有逐起点误差”的限制只适用于旧逐小时结果。原 NWP 文件没有逐条发行时间或预报版本，仍不能独立核验实际运营发行。
