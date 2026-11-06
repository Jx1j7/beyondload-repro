"""Post-hoc Chronos-2 past-weather-only arm on the frozen daily origins.

Future-known calendar is unchanged. Past NWP fields are supplied in past_df,
while future_df contains calendar only. Original four-arm files are untouched.
"""
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

from run_competition_chronos2_cpu import BASE, CONTEXT, HORIZON, REVISION, input_frames

BASELINE = BASE / "experiments/competition_aligned_cpu/chronos2"
OUT = BASE / "experiments/competition_aligned_cpu/chronos2_past_weather"
ORIGINS = BASE / "experiments/competition_aligned_cpu/candidate_origins.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_arm(pipe, site, selected, cov_cols, batch_origins=8):
    gs = selected.origin_index.astype(int).tolist()
    rows = []
    for start in range(0, len(gs), batch_origins):
        chunk = gs[start:start + batch_origins]
        past, future = input_frames(site, chunk, True, cov_cols)
        assert set(cov_cols).issubset(past.columns)
        future = future.drop(columns=cov_cols)
        assert not set(cov_cols).intersection(future.columns)
        predicted = pipe.predict_df(past, future_df=future,
                                    prediction_length=HORIZON,
                                    context_length=CONTEXT, batch_size=128,
                                    cross_learning=False)
        for j, g in enumerate(chunk):
            item = f"origin_{j}"
            out = predicted.loc[predicted.item_id.eq(item)].sort_values("timestamp")
            actual = site.power.iloc[g:g + HORIZON].to_numpy(dtype=float)
            times = site.timestamp.iloc[g:g + HORIZON].reset_index(drop=True)
            pred = out.predictions.to_numpy(dtype=float)
            assert len(out) == HORIZON and np.isfinite(pred).all()
            assert np.array_equal(pd.to_datetime(out.timestamp).to_numpy(), times.to_numpy())
            issue = times.iloc[0] - pd.Timedelta(hours=1)
            for h in range(HORIZON):
                rows.append(dict(origin_index=g, inferred_issue_time=issue,
                                 target_timestamp=times.iloc[h], horizon_step=h + 1,
                                 actual=actual[h], prediction=pred[h]))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    selected_all = pd.read_csv(ORIGINS).loc[lambda x: x.eligible]
    snapshot = snapshot_download("amazon/chronos-2", revision=REVISION, local_files_only=True)
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
        t0 = time.monotonic()
        points = run_arm(pipe, site, selected, cov_cols)
        assert len(points) == HORIZON * len(selected)
        baseline = pd.read_csv(BASELINE / f"{dataset}_calendar_points.csv")
        assert np.array_equal(points.origin_index, baseline.origin_index)
        assert np.array_equal(points.horizon_step, baseline.horizon_step)
        assert np.array_equal(points.actual, baseline.actual)
        path = OUT / f"{dataset}_past_weather_points.csv"
        points.to_csv(path, index=False)
        records.append(dict(dataset=dataset, origins=len(selected),
                            seconds=round(time.monotonic() - t0, 2),
                            file=path.name, sha256=sha(path)))
        print(dataset, records[-1]["seconds"], "s", flush=True)
    lines = ["# Chronos-2 past-weather-only CPU arm", "",
             f"- Started UTC: {started}",
             f"- Finished UTC: {datetime.now(timezone.utc).isoformat()}",
             f"- Model revision: `{REVISION}`",
             f"- Python: {sys.version.split()[0]}; Torch: {torch.__version__}",
             f"- Host: {platform.platform()}; Torch CPU threads: 4",
             f"- Source SHA-256: `{sha(Path(__file__))}`",
             f"- Candidate origins SHA-256: `{sha(ORIGINS)}`",
             "- Future-known calendar remains in `future_df`. Past NWP fields are "
             "only in past input; no future NWP fields are passed to `future_df`.",
             "- Forecasts are separately generated from the pinned checkpoint; "
             "no original four-arm prediction files are overwritten.", "",
             "| Dataset | Origins | CPU seconds | File | SHA-256 |",
             "|---|---:|---:|---|---|"]
    for r in records:
        lines.append(f"| {r['dataset']} | {r['origins']} | {r['seconds']} | `{r['file']}` | `{r['sha256']}` |")
    (OUT / "run_manifest.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
