"""Write manuscript Appendix B from the competition-aligned point forecasts."""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE / "experiments/competition_aligned_cpu"
OUT = BASE / "paper/appendix-B-competition-aligned.md"


def main() -> None:
    site = pd.read_csv(ROOT / "paired_per_site.csv")
    p = site.pivot(index="dataset", columns=["model", "arm"], values="mase")
    n = site.drop_duplicates("dataset").set_index("dataset").origins
    lines = ["# Appendix B. Competition-schedule-aligned paired point accuracy", "",
             "This post-hoc CPU analysis uses the same eligible daily 24-hour "
             "forecast windows for both models. Source weather records run from "
             "01:00 to next-day 00:00; the midnight issue time is inferred from "
             "the GEFCom2014 publication and is not an explicit file field. "
             "A window is retained only when its preceding 512-hour raw target "
             "context, next 24 raw target hours, and weather fields are complete. "
             "The old hourly-origin scores in Appendix A are a different protocol.", "",
             "LightGBM is site-trained with five seeds; scores below average seed "
             "scores. Chronos-2 is zero-shot at pinned local revision "
             "`29ec3766d36d6f73f0696f85560a422f50e8498c`. Both models receive "
             "future-known calendar in both arms. The NWP arms both receive the "
             "next 24 hours' weather fields; Chronos-2 additionally sees their "
             "past trajectory through its native interface. Wind is scored over "
             "all hours, solar over observed power >1% hours; both use a "
             "training-period 24-hour seasonal-naive denominator.", "",
             "| Site | Days | LightGBM calendar | Chronos-2 calendar | LightGBM + NWP | Chronos-2 + NWP | LightGBM lower by |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in p.index:
        r = p.loc[dataset]
        l = r[("LightGBM","weather")]
        c = r[("Chronos-2","weather")]
        lines.append(f"| {dataset} | {int(n[dataset])} | "
                     f"{r[('LightGBM','calendar')]:.3f} | "
                     f"{r[('Chronos-2','calendar')]:.3f} | {l:.3f} | {c:.3f} | "
                     f"{100*(c-l)/c:.1f}% |")
    lines += ["", "The percentage in the last column is relative to that site's "
              "Chronos-2 + NWP score. It is a descriptive effect, not a "
              "confidence bound. Wind zones share one source, and only three "
              "solar sites are available. Point forecasts, origin-level paired "
              "losses, exclusions, input hashes and execution manifests are in "
              "`experiments/competition_aligned_cpu/`.", ""]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
