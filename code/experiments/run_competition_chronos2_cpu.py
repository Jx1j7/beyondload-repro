"""Chronos-2 paired daily-origin CPU evaluation with explicit future covariates.

Uses competition-aligned origins from audit_competition_issue_schedule_cpu.py.
The two arms receive the same target history and known calendar; the weather arm
also receives past and future NWP. Original results.csv is never changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from chronos import Chronos2Pipeline

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "experiments/competition_aligned_cpu/chronos2"
ORIGINS = BASE / "experiments/competition_aligned_cpu/candidate_origins.csv"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
CONTEXT = 512
HORIZON = 24


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calendar(times: pd.Series) -> pd.DataFrame:
    t = pd.DatetimeIndex(times)
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * t.hour / 24),
        "hour_cos": np.cos(2 * np.pi * t.hour / 24),
        "dow_sin": np.sin(2 * np.pi * t.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * t.dayofweek / 7),
    })


def input_frames(site: pd.DataFrame, origins: list[int], weather: bool, cov_cols: list[str]):
    pasts, futures = [], []
    for index, g in enumerate(origins):
        item = f"origin_{index}"
        past = site.iloc[g-CONTEXT:g]
        future = site.iloc[g:g+HORIZON]
        p = pd.DataFrame({"item_id": item, "timestamp": past.timestamp.to_numpy(),
                          "target": past.power.to_numpy(dtype=float)})
        f = pd.DataFrame({"item_id": item, "timestamp": future.timestamp.to_numpy()})
        for name, values in calendar(past.timestamp).items():
            p[name] = values.to_numpy()
        for name, values in calendar(future.timestamp).items():
            f[name] = values.to_numpy()
        if weather:
            for col in cov_cols:
                p[col] = past[col].to_numpy(dtype=float)
                f[col] = future[col].to_numpy(dtype=float)
        pasts.append(p)
        futures.append(f)
    return pd.concat(pasts, ignore_index=True), pd.concat(futures, ignore_index=True)


def run_arm(pipe, site, selected, weather, cov_cols, batch_origins):
    gs = selected.origin_index.astype(int).tolist()
    rows = []
    for start in range(0, len(gs), batch_origins):
        chunk = gs[start:start+batch_origins]
        past, future = input_frames(site, chunk, weather, cov_cols)
        predicted = pipe.predict_df(past, future_df=future,
                                    prediction_length=HORIZON,
                                    context_length=CONTEXT, batch_size=128,
                                    cross_learning=False)
        for j, g in enumerate(chunk):
            item = f"origin_{j}"
            out = predicted.loc[predicted.item_id.eq(item)].sort_values("timestamp")
            actual = site.power.iloc[g:g+HORIZON].to_numpy(dtype=float)
            times = site.timestamp.iloc[g:g+HORIZON].reset_index(drop=True)
            pred = out.predictions.to_numpy(dtype=float)
            assert len(out) == HORIZON and np.isfinite(pred).all()
            assert np.array_equal(pd.to_datetime(out.timestamp).to_numpy(),
                                  times.to_numpy())
            issue = times.iloc[0] - pd.Timedelta(hours=1)
            for h in range(HORIZON):
                rows.append(dict(origin_index=g, inferred_issue_time=issue,
                                 target_timestamp=times.iloc[h], horizon_step=h+1,
                                 actual=actual[h], prediction=pred[h]))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("wind", "solar"))
    parser.add_argument("--zone", type=int)
    parser.add_argument("--batch-origins", type=int, default=8)
    args = parser.parse_args()
    assert args.batch_origins > 0
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    selected_all = pd.read_csv(ORIGINS)
    selected_all = selected_all.loc[selected_all.eligible].copy()
    if args.track:
        selected_all = selected_all.loc[selected_all.track.eq(args.track)]
    if args.zone is not None:
        selected_all = selected_all.loc[selected_all.zone.eq(args.zone)]
    assert not selected_all.empty
    snapshot = snapshot_download("amazon/chronos-2", revision=REVISION,
                                 local_files_only=True)
    pipe = Chronos2Pipeline.from_pretrained(snapshot)
    started = datetime.now(timezone.utc).isoformat()
    records = []
    for dataset, selected in selected_all.groupby("dataset", sort=True):
        track = selected.track.iloc[0]
        zone = int(selected.zone.iloc[0])
        site = pd.read_csv(BASE / f"data/processed/gefcom2014_{track}_clean.csv",
                           parse_dates=["timestamp"])
        site = site.loc[site.zone.eq(zone)].sort_values("timestamp").reset_index(drop=True)
        cov_cols = [c for c in site if c not in ("zone", "timestamp", "power")]
        assert len(site) >= selected.origin_index.max() + HORIZON
        for weather in (False, True):
            arm = "weather" if weather else "calendar"
            t0 = time.monotonic()
            points = run_arm(pipe, site, selected, weather, cov_cols,
                             args.batch_origins)
            assert len(points) == HORIZON * len(selected)
            path = OUT / f"{dataset}_{arm}_points.csv"
            points.to_csv(path, index=False)
            records.append(dict(dataset=dataset, arm=arm, origins=len(selected),
                                seconds=round(time.monotonic()-t0, 2),
                                file=path.name, sha256=sha(path)))
            print(f"{dataset} {arm}: {len(selected)} origins, "
                  f"{records[-1]['seconds']} s", flush=True)
    manifest = ["# Chronos-2 competition-aligned CPU manifest", "",
                f"- Started UTC: {started}",
                f"- Finished UTC: {datetime.now(timezone.utc).isoformat()}",
                f"- Model: amazon/chronos-2 at pinned revision `{REVISION}`",
                f"- Local snapshot: `{snapshot}`",
                f"- Python: {sys.version.split()[0]}; Torch: {torch.__version__}",
                f"- Host: {platform.platform()}; Torch CPU threads: 4",
                f"- Source script SHA-256: `{sha(Path(__file__))}`",
                f"- Candidate origins SHA-256: `{sha(ORIGINS)}`",
                "- Both arms: 512-hour target history and future-known calendar; "
                "weather arm additionally receives past and next-24-hour NWP "
                "through the Chronos-2 `future_df` API.",
                "- Daily issue time is inferred from competition schedule, not a "
                "source-file issue-time column. Original `results.csv` untouched.", "",
                "| Dataset | Arm | Origins | CPU seconds | File | SHA-256 |",
                "|---|---|---:|---:|---|---|"]
    for r in records:
        manifest.append(f"| {r['dataset']} | {r['arm']} | {r['origins']} | "
                        f"{r['seconds']} | `{r['file']}` | `{r['sha256']}` |")
    (OUT / "run_manifest.md").write_text("\n".join(manifest)+"\n")


if __name__ == "__main__":
    main()
