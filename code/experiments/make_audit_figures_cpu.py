"""Create evidence-bound manuscript figures from audited CSVs, without inference."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
FIG = BASE / "paper/figures"


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    per_site = pd.read_csv(BASE / "experiments/metric_audit/corrected_per_site.csv")
    pivot = per_site.pivot(index="dataset", columns="model", values="mase")
    complete = pivot.dropna(axis=1, how="any")
    assert complete.shape == (14, 5)
    ranks = complete.rank(axis=1).mean().sort_values()
    labels = {"lightgbm+cov": "LightGBM + target-time covariates",
              "chronos2": "Chronos-2", "patchtst+cov": "PatchTST",
              "chronos": "Chronos-Bolt", "itransformer+cov": "iTransformer"}
    fig, ax = plt.subplots(figsize=(8.1, 3.6))
    ys = np.arange(len(ranks))[::-1]
    ax.scatter(ranks.values, ys, s=70, color="#1b6c8a", zorder=3)
    for y, (model, rank) in zip(ys, ranks.items()):
        ax.text(rank + 0.09, y, f"{labels[model]}  {rank:.2f}", va="center", fontsize=9)
    ax.plot([1, 1 + 1.63], [5.0, 5.0], color="#b44833", lw=2)
    ax.plot([1, 1], [4.88, 5.12], color="#b44833", lw=2)
    ax.plot([1 + 1.63, 1 + 1.63], [4.88, 5.12], color="#b44833", lw=2)
    ax.text(1.815, 5.18, "Nominal Nemenyi CD = 1.63", ha="center", color="#b44833", fontsize=9)
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(-0.6, 5.65)
    ax.set_yticks([])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Mean rank across 14 site-datasets (lower is better)")
    ax.grid(axis="x", alpha=0.25)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    save(fig, "figure1_corrected_mean_ranks")

    paired = pd.read_csv(BASE / "experiments/information_match_cpu/per_site_comparison.csv")
    paired = paired[paired.track.isin(["wind", "solar"])].copy()
    paired["zone"] = paired.dataset.str.extract(r"zone(\d+)").astype(int)
    paired["order"] = paired.track.map({"wind": 0, "solar": 1})
    paired = paired.sort_values(["order", "zone"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.5))
    ys = np.arange(len(paired))[::-1]
    for y, row in zip(ys, paired.itertuples()):
        ax.plot([row.with_cov_mase, row.no_nwp_mase], [y, y],
                color="#9ba7ab", lw=1.5, zorder=1)
    ax.scatter(paired.with_cov_mase, ys, color="#146a85", s=42,
               label="LightGBM + target-time NWP", zorder=3)
    ax.scatter(paired.no_nwp_mase, ys, color="#c16c28", s=42,
               label="LightGBM without NWP", zorder=3)
    names = [f"{'Wind' if r.track == 'wind' else 'Solar'} {r.zone}"
             + (" *" if r.track == "wind" and r.zone == 3 else "")
             for r in paired.itertuples()]
    ax.set_yticks(ys, names)
    ax.set_xlabel("Corrected MASE (lower is better)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlim(0.25, 1.08)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(0, -0.13, "* Wind zone 3: cross-family origin sets differ.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    save(fig, "figure2_lightgbm_nwp_ablation")
    print(f"Wrote audited figures to {FIG}")


if __name__ == "__main__":
    main()
