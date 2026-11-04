"""CPU-only no-NWP LightGBM arm with vectorized, hourly rolling origins.

Uses the same training rows, features, hyperparameters, and recursive forecast
as run_trained.py, but batches origins per forecast horizon for CPU speed.
Writes only to experiments/information_match_cpu/ and never changes results.csv.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import platform
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import HORIZON, LOOKBACK, load_series, rolling_origins, split_indices

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "experiments/information_match_cpu"
LAGS = [1, 2, 3, 24, 25, 48, 72, 96, 120, 144, 168]


def features_1d(buf, timestamp):
    v = np.asarray(buf)
    return ([v[-lag] for lag in LAGS] +
            [v[-24:].mean(), v[-24:].std(), v[-168:].mean(),
             np.sin(2 * np.pi * timestamp.hour / 24),
             np.cos(2 * np.pi * timestamp.hour / 24),
             np.sin(2 * np.pi * timestamp.dayofweek / 7),
             np.cos(2 * np.pi * timestamp.dayofweek / 7)])


def train_model(values, timestamps, seed):
    train_end, test_start = split_indices(len(values))
    mu, sd = np.nanmean(values[:train_end]), np.nanstd(values[:train_end]) + 1e-8
    z = (values - mu) / sd

    def rows(start, end):
        x, y = [], []
        for i in range(max(LOOKBACK, start), end):
            if np.isnan(z[i]) or not np.isfinite(z[i - LOOKBACK:i]).all():
                continue
            x.append(features_1d(z[i - LOOKBACK:i], timestamps[i]))
            y.append(z[i])
        return np.asarray(x), np.asarray(y)

    x_train, y_train = rows(LOOKBACK, train_end)
    x_val, y_val = rows(train_end, test_start)
    reg = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05,
                            num_leaves=31, subsample=0.9,
                            colsample_bytree=0.9, random_state=seed, verbose=-1)
    reg.fit(x_train, y_train, eval_set=[(x_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return reg, z, mu, sd, test_start


def batch_rollout(reg, z, values, timestamps, test_start, mu, sd):
    candidate = [test_start + o for o in rolling_origins(len(values) - test_start, 1)]
    origins = np.array([g for g in candidate
                        if np.isfinite(values[g:g + HORIZON]).all()], dtype=int)
    buf = np.stack([z[g - LOOKBACK:g] for g in origins])
    forecast = np.empty((len(origins), HORIZON), dtype=float)
    for h in range(HORIZON):
        target_times = timestamps[origins + h]
        hours = np.asarray(target_times.hour)
        weekdays = np.asarray(target_times.dayofweek)
        x = np.column_stack(
            [buf[:, -lag] for lag in LAGS] +
            [buf[:, -24:].mean(axis=1), buf[:, -24:].std(axis=1),
             buf.mean(axis=1),
             np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
             np.sin(2 * np.pi * weekdays / 7),
             np.cos(2 * np.pi * weekdays / 7)])
        p = reg.predict(x)
        forecast[:, h] = p * sd + mu
        buf = np.column_stack((buf[:, 1:], p))
    actual = np.stack([values[g:g + HORIZON] for g in origins])
    return forecast, actual, origins


def main():
    OUT.mkdir(exist_ok=True)
    run_started = datetime.now(timezone.utc).isoformat()
    denom = pd.read_csv(BASE / "experiments/metric_audit/training_denominators.csv")
    denom_map = dict(zip(denom.dataset, denom.train_seasonal_naive_mae))
    rows = []
    for track, count in (("wind", 10), ("solar", 3), ("load", 1)):
        for zone in range(1, count + 1):
            timestamps, values = load_series(track, zone)
            dataset = f"gefcom2014_{track}_zone{zone}"
            for seed in range(5):
                started = time.monotonic()
                reg, z, mu, sd, test_start = train_model(values, timestamps, seed)
                pred, actual, origins = batch_rollout(
                    reg, z, values, timestamps, test_start, mu, sd)
                err = np.abs(pred - actual)
                if track == "solar":
                    mask = actual > 0.01
                    mae = float(err[mask].mean())
                else:
                    mae = float(err.mean())
                item = dict(dataset=dataset, track=track, model="lightgbm_no_nwp",
                            seed=seed, origins=len(origins), mae=mae,
                            mase=mae / denom_map[dataset],
                            best_iteration=reg.best_iteration_,
                            wall_s=round(time.monotonic() - started, 2),
                            run_started_utc=run_started,
                            row_recorded_utc=datetime.now(timezone.utc).isoformat())
                rows.append(item)
                pd.DataFrame(rows).to_csv(OUT / "lightgbm_nocov_step1.csv", index=False)
                print(f"{dataset} seed={seed} MASE={item['mase']:.3f} "
                      f"origins={len(origins)} {item['wall_s']:.1f}s", flush=True)
    run_finished = datetime.now(timezone.utc).isoformat()
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = ["# CPU no-NWP LightGBM run manifest", "",
                f"- Command: `.venv/bin/python code/experiments/run_lightgbm_nocov_cpu.py`",
                f"- Started (UTC): {run_started}",
                f"- Finished (UTC): {run_finished}",
                f"- Python: {sys.version.split()[0]}; LightGBM: {lgb.__version__}",
                f"- Host: {platform.platform()}",
                f"- Output rows: {len(rows)}; five seeds per 14 site-datasets",
                f"- Archived input results.csv SHA-256: {sha(BASE / 'experiments/results.csv')}",
                f"- Source script SHA-256: {sha(Path(__file__))}",
                f"- Output CSV SHA-256: {sha(OUT / 'lightgbm_nocov_step1.csv')}",
                "- Scope: post-hoc CPU arm, no NWP; future calendar features retained; "
                "original results.csv untouched.", ""]
    (OUT / "run_manifest.md").write_text("\n".join(manifest))


if __name__ == "__main__":
    main()
