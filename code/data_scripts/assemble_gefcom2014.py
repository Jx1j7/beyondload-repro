"""Assemble GEFCom2014 Wind/Solar/Load task files (1-15, cumulative) into full series.

Sources (official, ScienceDirect supplementary 1-s2.0-S0169207016000133-mmc1.zip):
  data/raw/gefcom2014_official/GEFCom2014 Data/

Logic: the Task-15 training files are cumulative (contain all history up to the
final forecast month); the "Solution to Task 15" files provide the last month of
realized values; solar NWP predictors are only issued per task month, so we union
predictors 1..15; wind NWP is inline in the task csvs and the final month's NWP
lives in the TaskExpVars15 zip.

Outputs (long format): data/processed/gefcom2014_{wind,solar,load}.csv
Citation: Hong, Pinson, Fan, Zareipour, Troccoli, Hyndman,
"Probabilistic energy forecasting: GEFCom2014 and beyond", IJF 2016.
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "gefcom2014_official" / "GEFCom2014 Data"
OUT = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def parse_ts(s: pd.Series, fmt: str) -> pd.Series:
    return pd.to_datetime(s, format=fmt)


def assemble_wind() -> pd.DataFrame:
    base = RAW / "GEFCom2014-W_V2" / "Wind"
    zpath = base / "Task 15" / "Task15_W_Zone1_10.zip"
    with zipfile.ZipFile(zpath) as z:
        parts = []
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(z.read(name)))
                parts.append(df)
    hist = pd.concat(parts, ignore_index=True)
    # NWP for the final month (Dec 2013)
    epath = base / "Task 15" / "TaskExpVars15_W_Zone1_10.zip"
    with zipfile.ZipFile(epath) as z:
        parts = [
            pd.read_csv(io.BytesIO(z.read(n))) for n in z.namelist() if n.lower().endswith(".csv")
        ]
    nwp = pd.concat(parts, ignore_index=True)
    sol = pd.read_csv(base / "Solution to Task 15" / "solution15_W.csv")
    last = sol.merge(nwp, on=["ZONEID", "TIMESTAMP"], how="left", validate="1:1")
    df = pd.concat([hist, last], ignore_index=True)
    df["timestamp"] = parse_ts(df["TIMESTAMP"], "%Y%m%d %H:%M")
    df = df.rename(columns={"ZONEID": "zone", "TARGETVAR": "power"})
    df = df[["zone", "timestamp", "power", "U10", "V10", "U100", "V100"]]
    df = df.sort_values(["zone", "timestamp"]).drop_duplicates(["zone", "timestamp"])
    return df


def assemble_solar() -> pd.DataFrame:
    base = RAW / "GEFCom2014-S_V2" / "Solar"
    hist = pd.read_csv(base / "Task 15" / "train15.csv")
    sol = pd.read_csv(base / "Solution to Task 15" / "Solution to Task 15.csv")
    power = pd.concat([hist, sol.rename(columns={"POWER": "POWER"})], ignore_index=True)
    power = power[["ZONEID", "TIMESTAMP", "POWER"]]
    nwps = []
    for k in range(1, 16):
        p = base / f"Task {k}" / f"predictors{k}.csv"
        nwps.append(pd.read_csv(p))
    nwp = pd.concat(nwps, ignore_index=True).drop_duplicates(["ZONEID", "TIMESTAMP"])
    nwp = nwp.drop(columns=["POWER"], errors="ignore")  # ground truth comes from train+solution only
    df = power.merge(nwp, on=["ZONEID", "TIMESTAMP"], how="left", validate="1:1")
    df["timestamp"] = parse_ts(df["TIMESTAMP"], "%Y%m%d %H:%M")
    df = df.rename(columns={"ZONEID": "zone", "POWER": "power"})
    df = df[["zone", "timestamp", "power"] + [c for c in df.columns if c.startswith("VAR")]]
    df = df.sort_values(["zone", "timestamp"]).drop_duplicates(["zone", "timestamp"])
    return df


def parse_load_ts(raw: pd.Series) -> pd.Series:
    """Parse unpadded MMDDYYYY H:MM stamps (e.g. '112001 1:00' = 2001-01-01 01:00).

    Month/day split is ambiguous for 3-digit MMDD (e.g. '112' = (1,12) or (11,2));
    resolved by hourly continuity with the previous row.
    """
    from datetime import datetime, timedelta

    def candidates(date_str: str):
        year = int(date_str[-4:])
        md = date_str[:-4]
        out = []
        for cut in range(1, len(md)):
            m, d = int(md[:cut]), int(md[cut:])
            try:
                out.append(datetime(year, m, d))
            except ValueError:
                pass
        return out

    parsed, prev = [], None
    for s in raw:
        date_str, hhmm = s.split()
        hour, minute = (int(x) for x in hhmm.split(":"))
        cands = [c.replace(hour=hour, minute=minute) for c in candidates(date_str)]
        if prev is None:
            dt = min(cands, key=lambda c: abs(c - datetime(2001, 1, 1)))
        else:
            want = prev + timedelta(hours=1)
            dt = min(cands, key=lambda c: abs(c - want))
        parsed.append(dt)
        prev = dt
    return pd.Series(pd.to_datetime(parsed), index=raw.index)


def assemble_load() -> pd.DataFrame:
    base = RAW / "GEFCom2014-L_V2" / "Load"
    # Task 1 train holds the full history (2001-01-01 .. 2010-10-01, zone 1 only);
    # tasks 2..15 are one-month increments.
    parts = [pd.read_csv(base / "Task 1" / "L1-train.csv")]
    for k in range(2, 16):
        parts.append(pd.read_csv(base / f"Task {k}" / f"L{k}-train.csv"))
    hist = pd.concat(parts, ignore_index=True)
    sol = pd.read_csv(base / "Solution to Task 15" / "solution15_L.csv")
    sol_t = pd.read_csv(base / "Solution to Task 15" / "solution15_L_temperature.csv")
    sol_t["TIMESTAMP"] = sol_t["date"].str.replace("/", "", regex=False) + " " + sol_t["hour"].astype(str) + ":00"
    sol_t["ZONEID"] = 1  # temperature solution covers zone 1 only
    sol_t = sol_t.drop(columns=["date", "hour", "LOAD"])  # LOAD from solution15_L is authoritative
    last = sol.merge(sol_t, on=["ZONEID", "TIMESTAMP"], how="left", validate="1:1")
    cols = hist.columns.tolist()
    last = last.reindex(columns=cols)
    df = pd.concat([hist, last], ignore_index=True)
    df["timestamp"] = parse_load_ts(df["TIMESTAMP"])
    df = df.rename(columns={"ZONEID": "zone", "LOAD": "load"})
    df = df[["zone", "timestamp", "load"] + [c for c in df.columns if c.startswith("w")]]
    df = df.sort_values(["zone", "timestamp"]).drop_duplicates(["zone", "timestamp"])
    return df


def report(name: str, df: pd.DataFrame, value_col: str) -> None:
    for zone, g in df.groupby("zone"):
        full_range = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="h")
        missing = len(full_range) - g["timestamp"].nunique()
        print(
            f"  {name} zone {zone}: {g['timestamp'].min()} .. {g['timestamp'].max()} "
            f"rows={len(g)} gaps={missing} {value_col}-null={g[value_col].isna().sum()} "
            f"range=[{g[value_col].min():.3f},{g[value_col].max():.3f}]"
        )


if __name__ == "__main__":
    for name, fn, col in [("wind", assemble_wind, "power"),
                          ("solar", assemble_solar, "power"),
                          ("load", assemble_load, "load")]:
        df = fn()
        print(f"== {name}: {len(df)} rows")
        report(name, df, col)
        df.to_csv(OUT / f"gefcom2014_{name}.csv", index=False)
        print(f"  -> {OUT / f'gefcom2014_{name}.csv'}")
