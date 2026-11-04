"""Recalculate scaled scores from archived forecast-error aggregates, without inference.

The archived results contain all-hour MAE and daylight MAE, so an all-hour wind
numerator and a solar-daylight numerator can be recovered without predictions.
This does not repair NWP availability, run provenance, or per-origin analyses.

Outputs are separate audit artifacts; the original results.csv is never changed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "experiments" / "results.csv"
OUT = BASE / "experiments" / "metric_audit"
PERIOD = 24


def load_target(track: str, zone: int) -> np.ndarray:
    path = BASE / "data" / "processed" / f"gefcom2014_{track}_clean.csv"
    data = pd.read_csv(path, parse_dates=["timestamp"])
    column = "load" if track == "load" else "power"
    series = (data.loc[data.zone == zone, ["timestamp", column]]
              .drop_duplicates("timestamp")
              .set_index("timestamp")[column]
              .sort_index().asfreq("h"))
    return series.to_numpy(dtype=float)


def training_denominator(values: np.ndarray, track: str) -> tuple[float, int]:
    """Training-sample seasonal naive MAE, period 24; solar daylight only."""
    train_end = int(len(values) * 0.7)
    current = values[PERIOD:train_end]
    previous = values[:train_end - PERIOD]
    valid = np.isfinite(current) & np.isfinite(previous)
    if track == "solar":
        valid &= current > 0.01
    if not valid.any():
        raise ValueError(f"No valid training pairs for {track}")
    denominator = float(np.mean(np.abs(current[valid] - previous[valid])))
    if denominator <= 0:
        raise ValueError(f"Non-positive denominator for {track}")
    return denominator, int(valid.sum())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(RESULTS)
    assert raw["MAE"].notna().all() and raw["MAE_daylight"].notna().all()
    assert np.allclose(raw["nMAE"], raw["MAE"]), "Archived all-hour nMAE and MAE differ"
    step1 = raw[raw.note.fillna("").str.contains(r"(?:^|;)\s*step=1(?:;|$)", regex=True)].copy()
    step1 = step1.drop_duplicates(["model", "dataset", "seed"], keep="last")

    site_rows = []
    for dataset in sorted(step1.dataset.unique()):
        _, track, zone_text = dataset.split("_", 2)
        zone = int(zone_text.removeprefix("zone"))
        values = load_target(track, zone)
        denominator, count = training_denominator(values, track)
        site_rows.append(dict(dataset=dataset, track=track, zone=zone,
                              train_seasonal_naive_mae=denominator,
                              train_pair_count=count,
                              train_end=int(len(values) * 0.7)))
    sites = pd.DataFrame(site_rows)
    scored = step1.merge(sites, on="dataset", validate="many_to_one")
    scored["audit_numerator_mae"] = np.where(scored.track.eq("solar"),
                                              scored.MAE_daylight, scored.MAE)
    scored["audit_standard_mase"] = (scored.audit_numerator_mae /
                                      scored.train_seasonal_naive_mae)
    assert np.isfinite(scored.audit_standard_mase).all()
    scored.to_csv(OUT / "corrected_step1_rows.csv", index=False)
    sites.to_csv(OUT / "training_denominators.csv", index=False)

    per_site = (scored.groupby(["dataset", "track", "model"], as_index=False)
                .agg(mase=("audit_standard_mase", "mean"),
                     seeds=("seed", "nunique")))
    per_site.to_csv(OUT / "corrected_per_site.csv", index=False)
    summary = (per_site.groupby(["track", "model"], as_index=False)
               .agg(mean_mase=("mase", "mean"), sites=("dataset", "nunique")))
    summary.to_csv(OUT / "corrected_cross_site.csv", index=False)

    pivot = per_site.pivot(index="dataset", columns="model", values="mase")
    complete = pivot.dropna(axis=1, how="any")
    ranks = complete.rank(axis=1)
    chi2, p = stats.friedmanchisquare(*[complete[col].to_numpy()
                                        for col in complete.columns])
    without_zone3 = complete.drop(index="gefcom2014_wind_zone3")
    chi2_without_zone3, p_without_zone3 = stats.friedmanchisquare(
        *[without_zone3[col].to_numpy() for col in without_zone3.columns])
    winners = pivot.loc[pivot.index.str.contains("wind|solar")].idxmin(axis=1)
    lines = ["# CPU 指标复算：独立审计结果", "",
             "原始 `experiments/results.csv` 未修改。本报告使用其逐次运行的已保存 MAE；没有重新训练或推理。",
             "", f"- 纳入：去重后 step=1 结果 {len(scored)} 行，{len(sites)} 个站点数据集。",
             "- 风电和负荷分子：归档的全时段 MAE；光伏分子：归档的实际功率 >1% 时段 MAE。",
             "- 分母：训练段内相隔 24 小时的季节朴素绝对误差均值；光伏分母限实际功率 >1% 时段。",
             "- 训练段终点：各站序列长度的 70%；缺失配对不计入分母。",
             "- 该复算无法验证未来 NWP 的发行时刻，也不能从汇总 MAE 恢复逐起点误差或置信区间。",
             "", "## 主结果敏感性", "",
             f"- 风光站点观测最优：{winners.value_counts().to_dict()}。",
             f"- Friedman（{len(complete.columns)} 模型 × {len(complete)} 数据集）：χ²={chi2:.2f}，名义 p={p:.3g}。",
             f"- 排除不同模型有效起点不一致的 wind zone 3 后，Friedman（{len(without_zone3.columns)} 模型 × {len(without_zone3)} 数据集）：χ²={chi2_without_zone3:.2f}，名义 p={p_without_zone3:.3g}。",
             "- 完整覆盖模型平均秩：" + "; ".join(f"{k} {v:.2f}" for k, v in ranks.mean().sort_values().items()),
             "", "## 跨站均值（标准化分母口径）", "",
             "| 轨道 | 模型 | 站点数 | 平均 MASE |", "|---|---|---:|---:|"]
    for _, row in summary.sort_values(["track", "mean_mase"]).iterrows():
        lines.append(f"| {row.track} | {row.model} | {int(row.sites)} | {row.mean_mase:.3f} |")
    lines += ["", "## 使用边界", "",
              "这是事后修正的指标口径，不是预注册原口径。`paper/manuscript-v2.md` 的表 3 已将其用作主表，并在方法中披露修正；旧口径留在原始 `results.csv`。风区同源且可能相关，Friedman p 值与 Nemenyi CD 基于独立块假设，只作名义描述，不能据此作总体显著性推断。该处理不能修复运行时间、NWP 发布时刻或 wind zone 3 的非共同起点问题。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:18]))
    print(f"Wrote audit artifacts to {OUT}")


if __name__ == "__main__":
    main()
