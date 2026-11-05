"""Exploratory within-site intervals from archived paired daily losses."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "experiments/competition_aligned_cpu"
SOURCE = ROOT / "paired_origin_losses.csv"
B = 5000
BLOCK = 7
SEED = 20261106


def weighted(values, counts):
    return np.sum(values * counts, axis=-1) / np.sum(counts, axis=-1)


def main():
    data = pd.read_csv(SOURCE)
    assert {"dataset", "origin_index", "count", "chronos2_calendar_mase",
            "chronos2_weather_mase", "lightgbm_calendar_mase",
            "lightgbm_weather_mase"} <= set(data.columns)
    sites = pd.read_csv(ROOT / "paired_per_site.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for site, group in data.groupby("dataset", sort=True):
        group = group.sort_values("origin_index")
        n = len(group)
        count = group["count"].to_numpy(dtype=float)
        assert np.all(count > 0)
        lc = group.lightgbm_calendar_mase.to_numpy()
        lw = group.lightgbm_weather_mase.to_numpy()
        cc = group.chronos2_calendar_mase.to_numpy()
        cw = group.chronos2_weather_mase.to_numpy()
        weather_gap = lw - cw
        calendar_gap = lc - cc
        interaction = weather_gap - calendar_gap
        # Circular blocks of consecutive *eligible origins*. Gaps in actual dates
        # mean these are sensitivity intervals, not a formal operational test.
        starts = rng.integers(0, n, size=(B, (n + BLOCK - 1) // BLOCK))
        index = (starts[:, :, None] + np.arange(BLOCK)) % n
        index = index.reshape(B, -1)[:, :n]
        values = {"weather_gap": weather_gap, "calendar_gap": calendar_gap,
                  "gap_change": interaction}
        row = {"dataset": site, "track": "solar" if "solar" in site else "wind",
               "origins": n, "count_total": int(count.sum())}
        for label, value in values.items():
            estimate = weighted(value, count)
            sampled = weighted(value[index], count[index])
            lo, hi = np.quantile(sampled, [0.025, 0.975])
            row.update({label: estimate, label + "_lo": lo, label + "_hi": hi})
        for model, arm, value in (("LightGBM", "calendar", lc),
                                  ("LightGBM", "weather", lw),
                                  ("Chronos-2", "calendar", cc),
                                  ("Chronos-2", "weather", cw)):
            archived = sites.loc[(sites.dataset == site) & (sites.model == model)
                                 & (sites.arm == arm), "mase"].iloc[0]
            assert abs(weighted(value, count) - archived) < 1e-12
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "paired_site_intervals.csv", index=False, float_format="%.12g")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    (ROOT / "paired_site_intervals_method.md").write_text(
        "# Exploratory within-site paired intervals\n\n"
        f"Source: `paired_origin_losses.csv` (SHA-256 `{source_hash}`). "
        "The `count` column is the number of scored target hours per origin; "
        "solar uses actual power > 0.01 and wind uses all 24 hours. "
        "All statistics are count-weighted means of corrected origin MASE. "
        "Each LightGBM origin loss is averaged over its five archived seeds before resampling.\n\n"
        f"For each site, a circular moving-block bootstrap resamples {BLOCK} "
        f"consecutive eligible origins at a time, {B:,} times, with NumPy seed {SEED}. "
        "The 2.5th and 97.5th percentiles are reported. Weather gap is "
        "LightGBM minus Chronos-2 with NWP; calendar gap is the analogous "
        "calendar-only difference; gap change is weather gap minus calendar gap. "
        "Intervals condition on each site's selected origin sequence and "
        "the fixed models and seeds. Eligible dates can have gaps, so adjacent "
        "selected origins are not always adjacent calendar days. Block length "
        "is a sensitivity choice, not a verified dependence horizon. No "
        "multiple-testing adjustment or cross-site population inference is made.\n")
    print(f"{len(result)} sites; source SHA-256 {source_hash}")
    print(result[["dataset", "weather_gap", "weather_gap_lo", "weather_gap_hi",
                  "gap_change"]].to_string(index=False, float_format=lambda x:f"{x:.3f}"))


if __name__ == "__main__":
    main()
