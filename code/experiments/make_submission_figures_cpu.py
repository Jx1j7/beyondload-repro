"""Render the article and supplementary figures from audited per-site CSV files."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/submission/figures"
BLUE = "#136f87"
ORANGE = "#b75a3a"
matplotlib.rcParams["ps.fonttype"] = 42


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "eps", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def primary():
    data = pd.read_csv(ROOT / "experiments/competition_aligned_cpu/paired_per_site.csv")
    weather = data.loc[data.arm.eq("weather")].copy()
    weather["zone"] = weather.dataset.str.extract(r"zone(\d+)").astype(int)
    weather["track_order"] = weather.track.map({"wind": 0, "solar": 1})
    weather = weather.sort_values(["track_order", "zone"])
    order = weather.drop_duplicates("dataset").dataset.tolist()
    assert len(order) == 13 and weather.shape[0] == 26
    scores = weather.pivot(index="dataset", columns="model", values="mase").loc[order]
    assert set(scores.columns) == {"LightGBM", "Chronos-2"}
    origins = weather.drop_duplicates("dataset").set_index("dataset").loc[order, "origins"]
    names = [f"{'Wind' if 'wind' in site else 'Solar'} {site.split('zone')[-1]}  (n={origins[site]})"
             for site in order]
    ys = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    for y, (_, row) in zip(ys, scores.iterrows()):
        ax.plot([row["LightGBM"], row["Chronos-2"]], [y, y],
                color="#aeb5ba", lw=1.2, zorder=1)
    ax.scatter(scores["LightGBM"], ys, color=BLUE, s=42, label="LightGBM + NWP", zorder=3)
    ax.scatter(scores["Chronos-2"], ys, color=ORANGE, s=42, label="Chronos-2 + NWP", zorder=3)
    ax.axhline(2.5, color="#d1d5d9", lw=1)
    ax.set_yticks(ys, names)
    ax.set_xlabel("Corrected MASE on common daily origins (lower is better)")
    ax.set_xlim(0.28, 0.92)
    ax.grid(axis="x", color="#e4e8ea", linewidth=0.7)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, "Figure_1_primary")


def ranks():
    data = pd.read_csv(ROOT / "experiments/metric_audit/corrected_per_site.csv")
    complete = data.pivot(index="dataset", columns="model", values="mase").dropna(axis=1)
    assert complete.shape == (14, 5)
    mean_ranks = complete.rank(axis=1).mean().sort_values()
    names = {"lightgbm+cov": "LightGBM + target-time covariates",
             "chronos2": "Chronos-2", "patchtst+cov": "PatchTST",
             "chronos": "Chronos-Bolt", "itransformer+cov": "iTransformer"}
    assert set(mean_ranks.index) == set(names)
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ys = np.arange(len(mean_ranks))[::-1]
    ax.scatter(mean_ranks.values, ys, s=58, color=BLUE, zorder=3)
    for y, (model, value) in zip(ys, mean_ranks.items()):
        ax.text(value + 0.08, y, f"{names[model]}  {value:.2f}", va="center", fontsize=9)
    ax.set_xlim(0.8, 5.3)
    ax.set_ylim(-0.6, 4.6)
    ax.set_yticks([])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Descriptive mean rank across 14 site-datasets (lower is better)")
    ax.grid(axis="x", color="#e4e8ea", linewidth=0.7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, "Figure_S1_retrospective_ranks")


def hourly_information():
    data = pd.read_csv(ROOT / "experiments/information_match_cpu/per_site_comparison.csv")
    data = data.loc[data.track.isin(["wind", "solar"])].copy()
    data["zone"] = data.dataset.str.extract(r"zone(\d+)").astype(int)
    data["track_order"] = data.track.map({"wind": 0, "solar": 1})
    data = data.sort_values(["track_order", "zone"])
    assert data.shape[0] == 13
    ys = np.arange(len(data))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.scatter(data.with_cov_mase, ys, color=BLUE, s=42,
               label="LightGBM + target-time NWP", zorder=3)
    ax.scatter(data.no_nwp_mase, ys, color=ORANGE, s=42,
               label="LightGBM without NWP", zorder=3)
    labels = [f"{'Wind' if row.track == 'wind' else 'Solar'} {row.zone}"
              for row in data.itertuples()]
    ax.set_yticks(ys, labels)
    ax.set_xlabel("Corrected MASE (lower is better)")
    ax.set_xlim(0.25, 1.08)
    ax.grid(axis="x", color="#e4e8ea", linewidth=0.7)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, "Figure_S2_hourly_information")


def horizon_profile():
    data = pd.read_csv(ROOT / "experiments/competition_aligned_cpu/horizon_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharey=False)
    for ax, track in zip(axes, ("wind", "solar")):
        frame = data.loc[data.track.eq(track)]
        for model, color in (("LightGBM", BLUE), ("Chronos-2", ORANGE)):
            for arm, style in (("calendar", "--"), ("weather", "-")):
                line = frame.loc[frame.model.eq(model) & frame.arm.eq(arm)].set_index("horizon").reindex(range(1, 25))
                ax.plot(line.index, line.mase, color=color, ls=style,
                        lw=1.8, label=f"{model}: {arm}")
        ax.set_title(f"{track.capitalize()} ({10 if track == 'wind' else 3} sites)")
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.grid(color="#e4e8ea", linewidth=.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Site-mean MASE at scored hours")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "Figure_2_horizon_profile")


if __name__ == "__main__":
    primary()
    ranks()
    hourly_information()
    horizon_profile()
    print(f"Wrote 4 figures in 4 formats to {OUT}")
