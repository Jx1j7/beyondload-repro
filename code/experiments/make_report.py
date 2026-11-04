"""Aggregate experiments/*.csv into a single readable Markdown report.

Usage: .venv/bin/python code/experiments/make_report.py
Output: experiments/报告.md  (regenerate any time; safe to overwrite)
"""
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
EXP = BASE / "experiments"


def fmt(df, cols=None, floats=4):
    if cols is None:
        cols = list(df.columns)
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:.{floats}f}", max_colwidth=60)


def main():
    parts = ["# beyondload 实验汇总报告\n",
             "自动生成，随时重跑：`.venv/bin/python code/experiments/make_report.py`。"
             "所有数字为**试点值**（快速配方或正式口径试点），不进论文；与正式值冲突以正式为准。\n"]

    res = pd.read_csv(EXP / "results.csv")
    parts.append(f"\n## 点预测总表（results.csv，{len(res)} 行）\n")
    for ds in res.dataset.unique():
        d = res[res.dataset == ds].sort_values("MASE_daylight")
        parts.append(f"\n### {ds}\n")
        show = d[["model", "MASE_daylight", "MAE_daylight", "seed", "time_s", "note"]]
        parts.append("```\n" + fmt(show) + "\n```")
        # seed stats for repeated rows
        grp = d.groupby("model")["MASE_daylight"].agg(["count", "mean", "std"])
        rep = grp[grp["count"] >= 2].sort_values("mean")
        if len(rep):
            parts.append("\n多种子模型（均值±标准差）：\n")
            for nm, row in rep.iterrows():
                parts.append(f"- {nm}: {row['mean']:.4f} ± {row['std']:.4f}（{int(row['count'])} 次）\n")

    for name, title in [("conformal_pilot.csv", "保形区间试点（RQ2）"),
                        ("tail_pilot.csv", "尾部/爬坡试点（RQ5）"),
                        ("routing_pilot.csv", "路由试点（RQ6）"),
                        ("efficiency_pilot.csv", "效率测量试点（RQ4，M1 CPU；正式版在租用 GPU 重测）")]:
        p = EXP / name
        if p.exists():
            df = pd.read_csv(p)
            parts.append(f"\n## {title}（{name}，{len(df)} 行）\n")
            parts.append("```\n" + fmt(df, list(df.columns)) + "\n```")

    parts.append("\n## 试点核心结论速览\n")
    parts.append("""1. **负荷 vs 风光反差**：负荷上零样本 TSFM 优于无协变量 LightGBM（0.70 vs 0.78）；风光上 lightgbm+cov 碾压（0.60/0.53 vs ≥0.69）——"Beyond load" 的立意有试点证据。
2. **协变量是决定性的**：LightGBM 光伏 0.999→0.608、风电 0.682→0.533；TSLib 深度模型只能吃过去协变量，无法吃到未来 NWP（架构差异，如实报告）。
3. **微调缩小但不消除**：TTM 光伏 1.006→0.965、风电 0.692→0.679。
4. **超参不敏感**：LightGBM 20 次随机搜索后 MASE 变化 ≤0.005，头名结论非调参伪影。
5. **RQ2**：光伏 CQR 可行（覆盖↑宽度↑）；风电加性 CQR 失效（区间盖满全域）→ 需自适应保形。
6. **RQ3 交叉点**：风电 ≈1–2 周、光伏 ≈1 个月训练数据后专用反超零样本。
7. **RQ5**：光伏爬坡检测 TSFM 0.19 ≪ 专用 0.41；风电全员 F1≈0（爬坡近不可预测）。
8. **RQ6**：朴素波动路由不优于最优单模型（风电 0.716 vs 0.694，负结果）——需更强路由特征。
""")

    out = EXP / "报告.md"
    out.write_text("".join(parts))
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
