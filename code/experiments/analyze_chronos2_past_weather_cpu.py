"""Score the new past-NWP-only Chronos-2 arm on frozen daily windows."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/competition_aligned_cpu"
PAST = BASE / "chronos2_past_weather"
denoms = pd.read_csv(ROOT / "experiments/metric_audit/training_denominators.csv").set_index("dataset")
original = pd.read_csv(BASE / "paired_per_site.csv")
rows = []
for site, group in original.groupby("dataset"):
    track = group.track.iloc[0]
    path = PAST / f"{site}_past_weather_points.csv"
    points = pd.read_csv(path)
    assert len(points) == 24 * int(group.origins.iloc[0])
    mask = points.actual.to_numpy(float) > .01 if track == "solar" else np.ones(len(points), bool)
    score = np.mean(np.abs(points.prediction.to_numpy(float)[mask] - points.actual.to_numpy(float)[mask])) / float(
        denoms.loc[site, "train_seasonal_naive_mae"])
    cal = float(group.loc[group.model.eq("Chronos-2") & group.arm.eq("calendar"), "mase"].iloc[0])
    full = float(group.loc[group.model.eq("Chronos-2") & group.arm.eq("weather"), "mase"].iloc[0])
    rows.append(dict(dataset=site, track=track, origins=int(group.origins.iloc[0]),
                     calendar_mase=cal, past_weather_mase=score, past_future_weather_mase=full,
                     past_gain_pct=100 * (1 - score / cal),
                     future_increment_pct=100 * (1 - full / score)))
site_scores = pd.DataFrame(rows)
site_scores.to_csv(BASE / "chronos2_weather_ablation_per_site.csv", index=False)
summary = site_scores.groupby("track", as_index=False).agg(
    sites=("dataset", "nunique"), calendar_mase=("calendar_mase", "mean"),
    past_weather_mase=("past_weather_mase", "mean"),
    past_future_weather_mase=("past_future_weather_mase", "mean"))
summary["past_gain_pct"] = 100 * (1 - summary.past_weather_mase / summary.calendar_mase)
summary["future_increment_pct"] = 100 * (1 - summary.past_future_weather_mase / summary.past_weather_mase)
summary.to_csv(BASE / "chronos2_weather_ablation_summary.csv", index=False)
print(summary.to_string(index=False))
