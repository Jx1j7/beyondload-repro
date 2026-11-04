"""Draw the same-future-NWP, same-origin per-site comparison."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "experiments/competition_aligned_cpu/paired_per_site.csv"
FIG = BASE / "paper/figures"


def main() -> None:
    frame = pd.read_csv(DATA)
    weather = frame.loc[frame.arm.eq("weather")].copy()
    weather["zone"] = weather.dataset.str.extract(r"zone(\d+)").astype(int)
    weather["track_order"] = weather.track.map({"wind": 0, "solar": 1})
    weather = weather.sort_values(["track_order", "zone"])
    pivot = weather.pivot(index="dataset", columns="model", values="mase")
    order = weather.drop_duplicates("dataset").dataset.tolist()
    pivot = pivot.loc[order]
    origins = weather.drop_duplicates("dataset").set_index("dataset").loc[order,"origins"]
    labels = []
    for dataset in order:
        track = "Wind" if "wind" in dataset else "Solar"
        zone = dataset.split("zone")[-1]
        labels.append(f"{track} {zone}  (n={origins[dataset]})")
    ys = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    for y, (_, row) in zip(ys, pivot.iterrows()):
        ax.plot([row["LightGBM"], row["Chronos-2"]], [y, y],
                color="#adb5bd", lw=1.3, zorder=1)
    ax.scatter(pivot["LightGBM"], ys, color="#136f87", s=45,
               label="LightGBM + NWP", zorder=3)
    ax.scatter(pivot["Chronos-2"], ys, color="#b75a3a", s=45,
               label="Chronos-2 + NWP", zorder=3)
    ax.axhline(2.5, color="#d1d5d9", lw=1)
    ax.set_yticks(ys, labels)
    ax.set_xlabel("Corrected MASE on matched daily origins (lower is better)")
    ax.set_xlim(0.28, 0.92)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="upper right", frameon=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"figure3_competition_aligned_weather.{ext}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Wrote competition-aligned figure 3")


if __name__ == "__main__":
    main()
