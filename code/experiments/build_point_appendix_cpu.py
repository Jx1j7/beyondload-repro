"""Build manuscript Appendix A from the independent corrected metric audit."""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "experiments/metric_audit/corrected_step1_rows.csv"
OUTPUT = BASE / "paper/appendix-A-corrected-point-accuracy.md"

MODELS = ["lightgbm+cov", "ttm", "timesfm", "moirai", "patchtst+cov",
          "timemixer+cov", "dlinear", "chronos2", "chronos",
          "itransformer+cov"]
LABELS = ["LightGBM+NWP", "TTM", "TimesFM", "Moirai-2", "PatchTST",
          "TimeMixer", "DLinear", "Chronos-2", "Chronos-Bolt", "iTransformer"]


def main() -> None:
    rows = pd.read_csv(INPUT)
    grouped = (rows.groupby(["track", "dataset", "model"], as_index=False)
               .agg(mean=("audit_standard_mase", "mean"),
                    sd=("audit_standard_mase", "std"),
                    seeds=("seed", "nunique")))
    assert len(rows) == 397 and grouped.dataset.nunique() == 14
    lookup = {(r.track, r.dataset, r.model): r for r in grouped.itertuples()}
    lines = ["# Appendix A. Corrected per-site point accuracy", "",
             "Generated from the archived 397 deduplicated `step=1` runs by "
             "`code/experiments/recalculate_metrics_cpu.py` and "
             "`code/experiments/build_point_appendix_cpu.py`. The original "
             "`experiments/results.csv` is unchanged.", "",
             "Scores are corrected MASE (lower is better): archived all-hour MAE "
             "for wind/load or observed-power >1% MAE for solar, divided by the "
             "site's 24-hour seasonal-naïve MAE in the training period. Each "
             "cell is the seed mean; five-seed cells also show the seed standard "
             "deviation. A dash means that no `step=1` result is archived for "
             "that combination. Seed variation is not a confidence interval "
             "over prediction origins or sites.", ""]
    for track, count in [("wind", 10), ("solar", 3), ("load", 1)]:
        lines += [f"## {track.title()} ({count} {'zones' if track != 'solar' else 'sites'})", "",
                  "| Site | " + " | ".join(LABELS) + " |",
                  "|---|" + "---:|" * len(MODELS)]
        for zone in range(1, count + 1):
            dataset = f"gefcom2014_{track}_zone{zone}"
            cells = []
            for model in MODELS:
                item = lookup.get((track, dataset, model))
                if item is None:
                    cells.append("—")
                elif item.seeds > 1:
                    cells.append(f"{item.mean:.3f} ± {item.sd:.3f} (n={item.seeds})")
                else:
                    cells.append(f"{item.mean:.3f}")
            suffix = " †" if track == "wind" and zone == 3 else ""
            lines.append(f"| {zone}{suffix} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["† Wind zone 3 has missing target intervals; the eligible-origin "
              "sets differ across model runners and context lengths. Its values "
              "are descriptive, not strictly paired model comparisons. The "
              "thirteen-site-dataset sensitivity test excluding this zone "
              "is reported in the main text and `experiments/metric_audit/report.md`.", "",
              "The complete per-run corrected values, site denominator, and "
              "site-level mean CSVs remain in `experiments/metric_audit/`. "
              "The archive has no per-origin forecasts or errors, so hourly "
              "paired confidence intervals and a common-origin zone-3 test "
              "cannot be reconstructed without additional inference.", ""]
    OUTPUT.write_text("\n".join(lines))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
