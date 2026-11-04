"""Summarize the post-hoc CPU no-NWP LightGBM experiment."""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "experiments/information_match_cpu"


def main():
    arm = pd.read_csv(OUT / "lightgbm_nocov_step1.csv")
    assert len(arm) == 70
    assert arm.groupby("dataset").seed.nunique().eq(5).all()
    archived = pd.read_csv(BASE / "experiments/metric_audit/corrected_per_site.csv")
    no_cov = (arm.groupby(["dataset", "track"], as_index=False)
              .agg(no_nwp_mase=("mase", "mean"),
                   seed_sd=("mase", "std"), origins=("origins", "first")))
    with_cov = archived[archived.model.eq("lightgbm+cov")][
        ["dataset", "mase"]].rename(columns={"mase": "with_cov_mase"})
    summary = no_cov.merge(with_cov, on="dataset", validate="one_to_one")
    summary["cov_reduction_pct"] = (100 * (summary.no_nwp_mase -
                                            summary.with_cov_mase) /
                                    summary.no_nwp_mase)
    summary.to_csv(OUT / "per_site_comparison.csv", index=False)

    means = summary.groupby("track")[["no_nwp_mase", "with_cov_mase"]].mean()
    fixed = {"wind": "ttm", "solar": "chronos2", "load": "chronos"}
    names = {"wind": "TTM", "solar": "Chronos-2", "load": "Chronos-Bolt"}
    lines = ["# Post-hoc CPU comparison: LightGBM with and without target-time covariates",
             "", "This is a new exploratory arm, separate from the original `results.csv` "
             "and the five-model main rank test. The 70 no-NWP rows cover "
             "14 site-datasets × 5 seeds at hourly origins. "
             "`run_lightgbm_nocov_cpu.py` reuses the original LightGBM training "
             "features and hyperparameters, with vectorized recursive prediction. "
             "On wind zone 1 and solar zone 1 at step=24, batched and original "
             "serial predictions agreed exactly (maximum absolute difference 0).",
             "", "The no-NWP arm still uses known future calendar features. "
             "It is a closer information comparison with target-history TSFMs, "
             "but feature sets and training methods are not identical. The "
             "with-covariate renewable arm uses retrospective target-time NWP "
             "whose issue time was not retained; its gain is not an estimate "
             "of deployable day-ahead skill.", "",
             "| Track | LightGBM no NWP | LightGBM with covariates | Fixed zero-shot comparator |",
             "|---|---:|---:|---:|"]
    for track in ("wind", "solar", "load"):
        model = fixed[track]
        tsfm = archived[(archived.track.eq(track)) &
                        (archived.model.eq(model))].mase.mean()
        lines.append(f"| {track} | {means.loc[track, 'no_nwp_mase']:.3f} | "
                     f"{means.loc[track, 'with_cov_mase']:.3f} | "
                     f"{names[track]} {tsfm:.3f} |")
    lines += ["", "Within LightGBM, adding target-time weather covariates "
              "reduces the corrected cross-site mean MASE by "
              f"{(1-means.loc['wind','with_cov_mase']/means.loc['wind','no_nwp_mase'])*100:.1f}% "
              "on wind and "
              f"{(1-means.loc['solar','with_cov_mase']/means.loc['solar','no_nwp_mase'])*100:.1f}% "
              "on solar. Without NWP, LightGBM and TTM are approximately "
              "tied on the wind cross-site mean; Chronos-2 is lower than "
              "LightGBM on each of the three solar sites. The load comparison "
              "uses temperature covariates rather than renewable NWP.", "",
              "| Site | No NWP | With covariates | Reduction (%) |",
              "|---|---:|---:|---:|"]
    for row in summary.sort_values(["track", "dataset"]).itertuples():
        lines.append(f"| {row.dataset} | {row.no_nwp_mase:.3f} | "
                     f"{row.with_cov_mase:.3f} | {row.cov_reduction_pct:.1f} |")
    lines += ["", "Wind zone 3 retains the unequal eligible-origin caveat "
              "for comparisons across model families. The two LightGBM arms "
              "use the same target-window filter in this zone. Results are "
              "descriptive; no per-origin paired uncertainty or operational "
              "NWP availability test is claimed.", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print(f"Wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
