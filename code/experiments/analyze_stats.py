"""Statistical analysis for the main experiment (protocol §2.5).

From experiments/results.csv (step=1 rows, deduped):
  1. per-dataset model scores (mean±std over seeds; zone1+zs single)
  2. Friedman test across the 13 zones (model ranks per zone)
  3. Nemenyi post-hoc: CD diagram numbers (rank means + critical difference)
  4. effect sizes: relative nMAE gap vs per-dataset best model (%)
Note: Diebold-Mariano needs per-origin loss series (runners don't persist
predictions yet) — flagged PENDING for the final protocol rerun with
--save_preds; Friedman/Nemenyi + effect sizes cover the frozen claims.

Usage: .venv/bin/python code/experiments/analyze_stats.py [--csv results.csv]
Output: experiments/stats_report.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[2]
# Nemenyi q_alpha=0.05 two-tailed, for k classifiers (standard table)
Q05 = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
       9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(BASE / "experiments/results.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    step1 = df[df.note.fillna("").str.contains("step=1")].copy()
    # dedupe: keep last occurrence (backfill/batched reruns supersede)
    step1 = step1.drop_duplicates(subset=["model", "dataset", "seed"], keep="last")

    # scores: mean over seeds per (model, dataset) using MASE_daylight (daylight convention)
    score = (step1.groupby(["model", "dataset"])["MASE_daylight"]
             .agg(["mean", "std", "count"]).reset_index())

    datasets = sorted(score.dataset.unique())
    models = sorted(score.model.unique())
    lines = ["# 主实验统计报告（协议 §2.5）\n",
             f"- 数据：{len(step1)} 行 step=1（去重后），{len(datasets)} 个数据集 × {len(models)} 个模型",
             "- 指标：MASE_daylight（多种子取均值）\n"]

    # rank matrix (model x dataset), lower score = better rank
    piv = score.pivot(index="model", columns="dataset", values="mean")
    piv = piv.dropna(axis=0, how="any")  # models present on all datasets only
    k, n = piv.shape
    lines.append(f"## Friedman 检验（k={k} 模型 × N={n} 数据集）\n")
    if n >= 3 and k >= 3:
        ranks = piv.rank(axis=0, method="average")
        chi2, p = stats.friedmanchisquare(*[piv.iloc[i].values for i in range(k)])
        lines.append(f"- Friedman χ² = {chi2:.2f}, p = {p:.3e}")
        q = Q05.get(k)
        cd = q * np.sqrt(k * (k + 1) / (6.0 * n)) if q else float("nan")
        mean_ranks = ranks.mean(axis=1).sort_values()
        lines.append(f"- Nemenyi CD(α=0.05) = {cd:.3f}（q={q}）\n")
        lines.append("### 平均秩（越小越好）\n")
        for m, r in mean_ranks.items():
            sig = " ✓组内最优" if r == mean_ranks.iloc[0] else ""
            lines.append(f"- {m}: {r:.2f}{sig}")
        lines.append("\n### 秩差超过 CD 的模型对（显著优于）\n")
        for i, m1 in enumerate(mean_ranks.index):
            for m2 in mean_ranks.index[i + 1:]:
                d = mean_ranks[m1] - mean_ranks[m2]  # m1 better if negative
                if d < -cd:
                    lines.append(f"- {m1} 显著优于 {m2}（Δ秩={d:.2f}）")

    lines.append("\n## 效应量：相对最优模型的差距（每数据集）\n")
    lines.append("| 数据集 | 最优模型 | 模型: 相对差% |")
    lines.append("|---|---|---|")
    for ds in datasets:
        sub = score[score.dataset == ds].sort_values("mean")
        best_m, best_v = sub.iloc[0]["model"], sub.iloc[0]["mean"]
        rel = ", ".join(f"{r.model}: +{(r['mean'] / best_v - 1) * 100:.0f}%"
                        for _, r in sub.iloc[1:5].iterrows())
        lines.append(f"| {ds.replace('gefcom2014_', '')} | {best_m} ({best_v:.3f}) | {rel} |")

    lines.append("\n## 各数据集完整成绩（mean±std）\n")
    for ds in datasets:
        sub = score[score.dataset == ds].sort_values("mean")
        lines.append(f"### {ds}")
        for _, r in sub.iterrows():
            sd = f"±{r['std']:.4f}" if pd.notna(r["std"]) and r["count"] > 1 else ""
            lines.append(f"- {r.model}: {r['mean']:.4f}{sd} (n={int(r['count'])})")
        lines.append("")

    lines.append("\n## 待补\n- Diebold–Mariano 两两检验：需逐起点损失序列，当前运行器未持久化预测；"
                 "最终协议版加 --save_preds 后补算（不影响 Friedman/Nemenyi 结论）。\n")

    out = BASE / "experiments" / "stats_report.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"...full -> {out}")


if __name__ == "__main__":
    main()
