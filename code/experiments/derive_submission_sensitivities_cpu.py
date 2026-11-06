"""Derive descriptive checks from archived point forecasts and paired losses."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/competition_aligned_cpu"
denom = pd.read_csv(ROOT / "experiments/metric_audit/training_denominators.csv").set_index("dataset")
sites = pd.read_csv(BASE / "paired_per_site.csv").dataset.unique()
rows = []
metric_rows = []
for site in sites:
    track = "solar" if "solar" in site else "wind"
    d = float(denom.loc[site, "train_seasonal_naive_mae"])
    for model, folder in (("Chronos-2", "chronos2"), ("LightGBM", "lightgbm")):
        for arm in ("calendar", "weather"):
            paths = sorted((BASE / folder).glob(f"{site}_{arm}*points.csv"))
            assert len(paths) == (1 if model == "Chronos-2" else 5), (site, arm, paths)
            seed_profiles = []
            seed_metrics = []
            for path in paths:
                frame = pd.read_csv(path)
                if track == "solar":
                    frame = frame.loc[frame.actual > .01]
                err = frame.actual - frame.prediction
                seed_metrics.append((err.abs().mean(), (err.pow(2).mean()) ** .5))
                profile = frame.assign(abs_error=(frame.actual - frame.prediction).abs()).groupby("horizon_step").agg(
                    mae=("abs_error", "mean"), points=("abs_error", "size"))
                seed_profiles.append(profile)
            metric_rows.append(dict(dataset=site, track=track, model=model, arm=arm,
                                    mae=sum(x[0] for x in seed_metrics)/len(seed_metrics),
                                    rmse=sum(x[1] for x in seed_metrics)/len(seed_metrics)))
            combined = pd.concat(seed_profiles, keys=range(len(seed_profiles)))
            for horizon, group in combined.groupby(level=1):
                assert group.points.nunique() == 1
                rows.append(dict(dataset=site, track=track, model=model, arm=arm,
                                 horizon=int(horizon), points=int(group.points.iloc[0]),
                                 mase=group.mae.mean()/d))
profile = pd.DataFrame(rows)
profile.to_csv(BASE / "horizon_site_profiles.csv", index=False)
summary = profile.groupby(["track", "model", "arm", "horizon"], as_index=False).agg(
    sites=("dataset", "nunique"), mase=("mase", "mean"),
    scored_points=("points", "sum"))
summary.to_csv(BASE / "horizon_summary.csv", index=False)
pd.DataFrame(metric_rows).to_csv(BASE / "capacity_error_per_site.csv", index=False)

losses = pd.read_csv(BASE / "paired_origin_losses.csv")
calendar = pd.read_csv(BASE / "candidate_origins.csv")[["dataset", "origin_index", "inferred_issue_time"]]
losses = losses.merge(calendar, on=["dataset", "origin_index"], validate="one_to_one")
wind = losses[losses.dataset.str.contains("wind")].copy()
wind.inferred_issue_time = pd.to_datetime(wind.inferred_issue_time)
first = wind.groupby("dataset").inferred_issue_time.min().max()
last = wind.groupby("dataset").inferred_issue_time.max().min()
subset = wind[wind.inferred_issue_time.between(first, last)]
assert subset.dataset.nunique() == 10
sens = []
for model, prefix in (("Chronos-2", "chronos2"), ("LightGBM", "lightgbm")):
    for arm in ("calendar", "weather"):
        col = f"{prefix}_{arm}_mase"
        by_site = subset.groupby("dataset").apply(
            lambda g: (g[col]*g["count"]).sum()/g["count"].sum(),
            include_groups=False)
        sens.append(dict(model=model, arm=arm, mean_mase=by_site.mean(),
                         sites=by_site.size, min_origins=subset.groupby("dataset").size().min(),
                         max_origins=subset.groupby("dataset").size().max(),
                         first_day=first.date(), last_day=last.date()))
pd.DataFrame(sens).to_csv(BASE / "common_date_sensitivity.csv", index=False)
print("Wrote horizon profiles, summary, and common-date sensitivity")
