"""Rescore the same daily forecasts under the original and corrected MASE rules.

This is a post-hoc sensitivity analysis, not a replay of archived hourly runs.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/experiments"))
from common import seasonal_naive_denom  # noqa: E402

BASE = ROOT / "experiments/competition_aligned_cpu"
corrected = pd.read_csv(BASE / "paired_per_site.csv")
new_denom = pd.read_csv(ROOT / "experiments/metric_audit/training_denominators.csv").set_index("dataset")
rows = []
for site, group in corrected.groupby("dataset"):
    track = group.track.iloc[0]
    zone = int(site.rsplit("zone", 1)[1])
    source = pd.read_csv(ROOT / f"data/processed/gefcom2014_{track}_clean.csv")
    values = source.loc[source.zone.eq(zone), "power"].to_numpy(float)
    old_denom = seasonal_naive_denom(values, int(len(values) * .8), step=24)
    assert np.isfinite(old_denom) and old_denom > 0
    for r in group.itertuples():
        folder = "chronos2" if r.model == "Chronos-2" else "lightgbm"
        paths = sorted((BASE / folder).glob(f"{site}_{r.arm}*points.csv"))
        assert len(paths) == r.seeds
        old_seed_scores = []
        for path in paths:
            points = pd.read_csv(path)
            mask = points.actual.to_numpy(float) > .01
            assert mask.any()
            old_seed_scores.append(float(np.mean(np.abs(
                points.prediction.to_numpy(float)[mask] - points.actual.to_numpy(float)[mask])) / old_denom))
        rows.append(dict(dataset=site, track=track, model=r.model, arm=r.arm,
                         origins=r.origins, corrected_mase=r.mase,
                         reconstructed_old_mase=np.mean(old_seed_scores),
                         old_test_denominator=old_denom,
                         corrected_train_denominator=new_denom.loc[site, "train_seasonal_naive_mae"]))
site_scores = pd.DataFrame(rows)
site_scores.to_csv(BASE / "metric_definition_sensitivity_per_site.csv", index=False)
means = site_scores.groupby(["track", "model", "arm"], as_index=False).agg(
    corrected_mase=("corrected_mase", "mean"),
    reconstructed_old_mase=("reconstructed_old_mase", "mean"),
    sites=("dataset", "nunique"))
means.to_csv(BASE / "metric_definition_sensitivity_summary.csv", index=False)
for arm in ("calendar", "weather"):
    sub = site_scores.loc[site_scores.arm.eq(arm)]
    old = sub.pivot(index="dataset", columns="model", values="reconstructed_old_mase")
    new = sub.pivot(index="dataset", columns="model", values="corrected_mase")
    flips = old.index[((old["LightGBM"] < old["Chronos-2"]) !=
                       (new["LightGBM"] < new["Chronos-2"]))].tolist()
    print(arm, "rank flips:", flips)
print(means.to_string(index=False))
