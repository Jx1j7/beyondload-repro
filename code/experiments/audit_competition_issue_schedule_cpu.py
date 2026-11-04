"""Audit GEFCom daily NWP blocks and common, competition-aligned test origins.

The source files omit an issue-time column. The publication describes midnight
issuance, and each source day spans valid hours 01:00 through next-day 00:00.
Thus each 24-hour block is assigned an *inferred* midnight issue time. The
audit does not claim that an individual forecast's vintage is independently
recorded in the files.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from common import HORIZON, LOOKBACK, rolling_origins, split_indices

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "experiments/competition_aligned_cpu"
CONTEXT = 512


def site_frame(track: str, zone: int, cleaned: bool) -> pd.DataFrame:
    suffix = "_clean" if cleaned else ""
    path = BASE / f"data/processed/gefcom2014_{track}{suffix}.csv"
    table = pd.read_csv(path, parse_dates=["timestamp"])
    return (table.loc[table.zone.eq(zone)].drop(columns="zone")
            .drop_duplicates("timestamp").set_index("timestamp")
            .sort_index().asfreq("h"))


def build() -> pd.DataFrame:
    rows = []
    for track, zones in (("wind", 10), ("solar", 3)):
        for zone in range(1, zones + 1):
            clean = site_frame(track, zone, True)
            raw = site_frame(track, zone, False)
            assert clean.index.equals(raw.index)
            assert clean.index.min().hour == 1 and clean.index.max().hour == 0
            source_day_counts = (clean.index - pd.Timedelta(hours=1)).normalize().value_counts()
            assert source_day_counts.eq(HORIZON).all(), (track, zone)
            target = clean.power.to_numpy(dtype=float)
            raw_target = raw.power.to_numpy(dtype=float)
            weather = clean.drop(columns="power").to_numpy(dtype=float)
            _, test_start = split_indices(len(clean))
            for o in rolling_origins(len(clean) - test_start, 1):
                g = test_start + o
                if clean.index[g].hour != 1:
                    continue
                block = clean.index[g:g + HORIZON]
                inferred_issue = (block - pd.Timedelta(hours=1)).normalize()
                assert len(block) == HORIZON and inferred_issue.nunique() == 1
                assert block[-1].hour == 0
                raw_context = bool(np.isfinite(raw_target[g-CONTEXT:g]).all())
                raw_future = bool(np.isfinite(raw_target[g:g+HORIZON]).all())
                clean_context = bool(np.isfinite(target[g-CONTEXT:g]).all())
                clean_future = bool(np.isfinite(target[g:g+HORIZON]).all())
                weather_context = bool(np.isfinite(weather[g-CONTEXT:g]).all())
                weather_future = bool(np.isfinite(weather[g:g+HORIZON]).all())
                rows.append(dict(dataset=f"gefcom2014_{track}_zone{zone}",
                                 track=track, zone=zone, origin_index=g,
                                 inferred_issue_time=inferred_issue[0],
                                 target_start=block[0], target_end=block[-1],
                                 raw_context_complete=raw_context,
                                 raw_future_complete=raw_future,
                                 clean_context_complete=clean_context,
                                 clean_future_complete=clean_future,
                                 weather_context_complete=weather_context,
                                 weather_future_complete=weather_future,
                                 eligible=all((raw_context, raw_future, clean_context,
                                               clean_future, weather_context, weather_future))))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frame = build()
    frame.to_csv(OUT / "candidate_origins.csv", index=False)
    counts = (frame.groupby(["track", "dataset"], as_index=False)
              .agg(candidates=("eligible", "size"), eligible=("eligible", "sum"),
                   raw_context_complete=("raw_context_complete", "sum"),
                   raw_future_complete=("raw_future_complete", "sum")))
    counts.to_csv(OUT / "availability_by_site.csv", index=False)
    sha = hashlib.sha256((OUT / "candidate_origins.csv").read_bytes()).hexdigest()
    lines = ["# GEFCom competition-aligned origin audit", "",
             "The original competition paper specifies daily midnight NWP issuance. "
             "The local wind/solar weather records run from 01:00 through next-day 00:00 "
             "in contiguous 24-hour blocks (731 days per wind site and 821 days "
             "per solar site), so the issue time below is inferred as "
             "midnight before the 01:00 valid hour. The files contain no explicit "
             "issue-time or forecast-vintage field; this is a competition-schedule "
             "alignment, not independently timestamped operational NWP.", "",
             "For each candidate, the 24-hour target and preceding 512-hour raw "
             "target context must be complete. This removes windows touched by "
             "the cleaning script's bidirectional interpolation and gives all "
             "compared models identical test origins. Weather fields must be finite "
             "over context and horizon.", "",
             f"- Candidate daily origins: {len(frame)}.",
             f"- Eligible common origins: {int(frame.eligible.sum())}.",
             f"- Candidate-origins CSV SHA-256: `{sha}`.", "",
             "| Track | Site | Candidates | Eligible | Raw context complete | Raw target complete |",
             "|---|---|---:|---:|---:|---:|"]
    for r in counts.itertuples():
        lines.append(f"| {r.track} | {r.dataset} | {r.candidates} | {r.eligible} | "
                     f"{r.raw_context_complete} | {r.raw_future_complete} |")
    lines += ["", "Reference: Hong et al., *International Journal of Forecasting* 32 "
              "(2016), §§6–7: https://robjhyndman.com/papers/gefcom2014.pdf", ""]
    (OUT / "availability_report.md").write_text("\n".join(lines))
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
