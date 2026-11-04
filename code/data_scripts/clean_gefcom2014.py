"""Clean GEFCom2014 assembled series per docs/05 §2.1 (pre-registered rules).

Rules:
  - missing runs of <= 3 hourly steps -> linear interpolation
  - missing runs of > 3 steps -> left as NaN (segment excluded from training/eval)
  - solar night zeros kept (no removal); metrics restricted to daylight elsewhere
  - capacity normalization: GEFCom2014 wind/solar power is already issued in
    [0, 1] of installed capacity by the competition, kept as-is
  - load: 2001-2004 has no load values by design (backcast period of the
    competition); series is trimmed to 2005-01-01 onward

Input : data/processed/gefcom2014_{wind,solar,load}.csv  (long format)
Output: data/processed/gefcom2014_{wind,solar,load}_clean.csv + cleaning_report.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
IN = BASE / "data" / "processed"
MAX_INTERP = 3  # hours


def flag_run_lengths(mask: pd.Series) -> pd.Series:
    """Return run length for each True element of mask."""
    grp = (mask != mask.shift()).cumsum()
    return mask.groupby(grp).transform("sum").where(mask)


def clean_series(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    df = df.set_index("timestamp")
    stats = {}
    for col in value_cols:
        s = df[col].astype(float)
        full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
        missing = full.isna()
        runs = flag_run_lengths(missing)
        short = missing & (runs <= MAX_INTERP)
        long = missing & (runs > MAX_INTERP)
        filled = full.interpolate(method="time", limit=MAX_INTERP, limit_area="inside")
        # re-apply: only fills where runs are short; long runs back to NaN
        out = filled.where(~long)
        if col == "power":  # competition artifacts exceed 1 (e.g. zone 3 max 1.004)
            out = out.clip(0, 1)
        df[col] = out.reindex(df.index)
        stats[col] = dict(total_missing=int(missing.sum()),
                          interpolated=int(short.sum()),
                          dropped=int(long.sum()))
    return df.reset_index(), stats


def main() -> None:
    report = []
    for track in ["wind", "solar", "load"]:
        df = pd.read_csv(IN / f"gefcom2014_{track}.csv", parse_dates=["timestamp"])
        if track == "load":
            before = len(df)
            df = df[df["timestamp"] >= "2005-01-01"].copy()
            report.append(f"[load] trimmed 2001-2004 backcast period: {before} -> {len(df)} rows")
        value_cols = [c for c in df.columns
                      if c not in ("zone", "timestamp") and pd.api.types.is_numeric_dtype(df[c])]
        out_parts = []
        for zone, g in df.groupby("zone"):
            g = g.drop(columns=["zone"]).sort_values("timestamp").drop_duplicates("timestamp")
            cleaned, stats = clean_series(g, value_cols)
            cleaned.insert(0, "zone", zone)
            out_parts.append(cleaned)
            worst = max(stats.items(), key=lambda kv: kv[1]["total_missing"])
            report.append(f"[{track} zone {zone}] worst col {worst[0]}: "
                          f"{worst[1]['total_missing']} missing -> {worst[1]['interpolated']} interp, "
                          f"{worst[1]['dropped']} kept-NaN (> {MAX_INTERP}h runs)")
        out = pd.concat(out_parts, ignore_index=True)
        path = IN / f"gefcom2014_{track}_clean.csv"
        out.to_csv(path, index=False)
        report.append(f"[{track}] -> {path} ({len(out)} rows)")
    text = "\n".join(report)
    (IN / "cleaning_report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
