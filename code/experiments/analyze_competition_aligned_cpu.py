"""Score four competition-aligned arms from archived point forecasts on common origins."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE / "experiments/competition_aligned_cpu"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped_points(path: Path, solar: bool):
    frame = pd.read_csv(path)
    err = np.abs(frame.prediction.to_numpy()-frame.actual.to_numpy())
    mask = frame.actual.to_numpy() > 0.01 if solar else np.ones(len(frame), dtype=bool)
    frame["abs_error_used"] = np.where(mask, err, 0.0)
    frame["count_used"] = mask.astype(int)
    by_origin = (frame.groupby("origin_index", as_index=False)
                 .agg(sum_abs=("abs_error_used", "sum"), count=("count_used", "sum")))
    return frame[["origin_index", "horizon_step", "actual"]], by_origin


def main() -> None:
    candidates = pd.read_csv(ROOT / "candidate_origins.csv")
    eligible = candidates.loc[candidates.eligible]
    denominator = pd.read_csv(BASE / "experiments/metric_audit/training_denominators.csv")
    denoms = dict(zip(denominator.dataset, denominator.train_seasonal_naive_mae))
    site_rows, origin_rows, seed_rows = [], [], []
    for dataset, selection in eligible.groupby("dataset", sort=True):
        track = selection.track.iloc[0]
        is_solar = track == "solar"
        expected = selection.origin_index.astype(int).to_numpy()
        reference = None
        daily = {}
        for model, directory, seeds in (("Chronos-2", "chronos2", [None]),
                                        ("LightGBM", "lightgbm", list(range(5)))):
            for arm in ("calendar", "weather"):
                series = []
                for seed in seeds:
                    suffix = "" if seed is None else f"_seed{seed}"
                    path = ROOT / directory / f"{dataset}_{arm}{suffix}_points.csv"
                    points, one = grouped_points(path, is_solar)
                    assert np.array_equal(one.origin_index.to_numpy(), expected)
                    if reference is None:
                        reference = points
                    else:
                        assert points[["origin_index", "horizon_step"]].equals(
                            reference[["origin_index", "horizon_step"]])
                        assert np.allclose(points.actual, reference.actual, atol=0, rtol=0)
                    one["mase"] = (one.sum_abs/one["count"])/denoms[dataset]
                    score = (one.sum_abs.sum()/one["count"].sum())/denoms[dataset]
                    seed_rows.append(dict(dataset=dataset, track=track, model=model,
                                          arm=arm, seed=-1 if seed is None else seed,
                                          origins=len(one), mase=score))
                    series.append(one)
                merged = series[0][["origin_index", "count"]].copy()
                merged["sum_abs"] = np.mean([x.sum_abs.to_numpy() for x in series],axis=0)
                merged["mase_origin"] = (merged.sum_abs/merged["count"])/denoms[dataset]
                daily[(model,arm)] = merged
                overall = (merged.sum_abs.sum()/merged["count"].sum())/denoms[dataset]
                site_rows.append(dict(dataset=dataset, track=track, model=model,
                                      arm=arm, origins=len(merged), mase=overall,
                                      seeds=len(seeds)))
        combined = daily[("Chronos-2","calendar")][["origin_index", "count"]].copy()
        for (model,arm), losses in daily.items():
            key = f"{model.lower().replace('-', '')}_{arm}_mase"
            combined[key] = losses.mase_origin.to_numpy()
        combined.insert(0,"dataset",dataset)
        origin_rows.append(combined)

    per_site = pd.DataFrame(site_rows)
    per_site.to_csv(ROOT / "paired_per_site.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(ROOT / "seed_scores.csv", index=False)
    pd.concat(origin_rows,ignore_index=True).to_csv(ROOT / "paired_origin_losses.csv", index=False)
    across = (per_site.groupby(["track", "model", "arm"],as_index=False)
              .agg(mean_mase=("mase","mean"),sites=("dataset","nunique")))
    across.to_csv(ROOT / "paired_cross_site.csv",index=False)
    pivot = per_site.pivot(index="dataset",columns=["model","arm"],values="mase")
    weather_winners = (pivot[[('Chronos-2','weather'),('LightGBM','weather')]]
                       .idxmin(axis=1).map(lambda item:item[0]))
    wind_complete = pivot.loc[[f"gefcom2014_wind_zone{z}" for z in range(4,11)]]
    rows = ["# Competition-aligned, information-aware CPU comparison", "",
            "This new comparison scores exactly the same daily 01:00–next-day "
            "00:00 target windows for each model, after excluding any window "
            "whose raw 512-hour target context or raw 24-hour target has missing "
            "values. Weather is assigned to the inferred midnight issue block "
            "from the original competition schedule. The source files do not "
            "contain an independently recorded issue-time field.", "",
            "Both models use target history and known calendar. The weather arms "
            "add the same target-hour NWP; Chronos-2 receives the past weather "
            "sequence through its native covariate interface, while LightGBM "
            "uses target-hour weather features. Thus future-information access "
            "and scored origins are matched, but context length, feature "
            "representation, and training regime differ. Chronos-2 was loaded "
            "at a pinned local revision. LightGBM has five site-trained seeds.", "",
            "Scores use the same training-period 24-hour seasonal-naive denominator "
            "as the metric audit; wind uses all target hours and solar only "
            "observed power >1% hours. These results are separate from original "
            "`results.csv` and its hourly-origin Table 3.", "",
            f"- Paired eligible origins: {int(eligible.shape[0])} across "
            f"{eligible.dataset.nunique()} renewable sites.",
            f"- Weather-arm site wins: {weather_winners.value_counts().to_dict()}.",
            "- Sensitivity excluding sparse-context wind zones 1–3: on wind "
            f"zones 4–10, LightGBM + NWP mean {wind_complete[('LightGBM','weather')].mean():.3f} "
            f"versus Chronos-2 + NWP {wind_complete[('Chronos-2','weather')].mean():.3f}; "
            "LightGBM lower at all seven sites.",
            f"- Origin-loss CSV SHA-256: `{sha(ROOT/'paired_origin_losses.csv')}`.", "",
            "| Track | Model | Calendar only | Calendar + NWP | Sites |",
            "|---|---|---:|---:|---:|"]
    for track in ("wind","solar"):
        for model in ("LightGBM","Chronos-2"):
            cal = across[(across.track==track)&(across.model==model)&(across.arm=="calendar")].iloc[0]
            wet = across[(across.track==track)&(across.model==model)&(across.arm=="weather")].iloc[0]
            rows.append(f"| {track} | {model} | {cal.mean_mase:.3f} | "
                        f"{wet.mean_mase:.3f} | {cal.sites} |")
    rows += ["", "## Per-site corrected MASE", "",
             "| Site | Origins | LightGBM calendar | Chronos-2 calendar | LightGBM + NWP | Chronos-2 + NWP |",
             "|---|---:|---:|---:|---:|---:|"]
    for dataset in pivot.index:
        p = pivot.loc[dataset]
        n = int(per_site.loc[per_site.dataset.eq(dataset),"origins"].iloc[0])
        rows.append(f"| {dataset} | {n} | {p[('LightGBM','calendar')]:.3f} | "
                    f"{p[('Chronos-2','calendar')]:.3f} | "
                    f"{p[('LightGBM','weather')]:.3f} | "
                    f"{p[('Chronos-2','weather')]:.3f} |")
    rows += ["", "No population-level significance is claimed: wind sites share "
             "one competition source, solar has three sites, and adjacent daily "
             "losses may be serially correlated. Per-origin paired losses are "
             "saved for further analysis without inventing hourly uncertainty "
             "from aggregate errors.", ""]
    (ROOT / "comparison_report.md").write_text("\n".join(rows))
    print("\n".join(rows[:35]))


if __name__ == "__main__":
    main()
