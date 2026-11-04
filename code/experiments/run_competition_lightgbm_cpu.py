"""LightGBM paired daily-origin CPU evaluation on the GEFCom issue schedule.

Training setup matches the original LightGBM runner. Both arms have target lags
and calendar features; the weather arm additionally uses NWP at target hours.
Both are scored on the same interpolation-free 24-hour test windows used by
run_competition_chronos2_cpu.py. Original results.csv is never changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import HORIZON, LOOKBACK, split_indices

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "experiments/competition_aligned_cpu/lightgbm"
ORIGINS = BASE / "experiments/competition_aligned_cpu/candidate_origins.csv"
LAGS = [1, 2, 3, 24, 25, 48, 72, 96, 120, 144, 168]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_features(buf, timestamp, cov_now):
    v = np.asarray(buf)
    features = ([v[-lag] for lag in LAGS] +
                [v[-24:].mean(), v[-24:].std(), v[-168:].mean(),
                 np.sin(2*np.pi*timestamp.hour/24),
                 np.cos(2*np.pi*timestamp.hour/24),
                 np.sin(2*np.pi*timestamp.dayofweek/7),
                 np.cos(2*np.pi*timestamp.dayofweek/7)])
    if cov_now is not None:
        features += list(np.asarray(cov_now))
    return features


def train(site: pd.DataFrame, seed: int, weather: bool):
    values = site.power.to_numpy(dtype=float)
    ts = pd.DatetimeIndex(site.timestamp)
    train_end, test_start = split_indices(len(values))
    mu, sd = np.nanmean(values[:train_end]), np.nanstd(values[:train_end])+1e-8
    z = (values-mu)/sd
    cov = None
    if weather:
        raw = site.drop(columns=["zone", "timestamp", "power"])
        cov_mu = raw.iloc[:train_end].mean()
        cov_sd = raw.iloc[:train_end].std().replace(0, 1.0)+1e-8
        cov = ((raw-cov_mu)/cov_sd).to_numpy(dtype=float)

    def rows(start: int, end: int):
        X, y = [], []
        for i in range(max(LOOKBACK, start), end):
            if np.isnan(z[i]) or not np.isfinite(z[i-LOOKBACK:i]).all():
                continue
            X.append(one_features(z[i-LOOKBACK:i], ts[i],
                                  None if cov is None else cov[i]))
            y.append(z[i])
        return np.asarray(X), np.asarray(y)

    X_train, y_train = rows(LOOKBACK, train_end)
    X_val, y_val = rows(train_end, test_start)
    reg = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05,
                            num_leaves=31, subsample=0.9,
                            colsample_bytree=0.9, random_state=seed, verbose=-1)
    reg.fit(X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return reg, z, cov, mu, sd


def forecast(reg, site, gs, z, cov, mu, sd):
    values = site.power.to_numpy(dtype=float)
    ts = pd.DatetimeIndex(site.timestamp)
    origins = np.asarray(gs, dtype=int)
    buf = np.stack([z[g-LOOKBACK:g] for g in origins])
    pred = np.empty((len(origins), HORIZON), dtype=float)
    for h in range(HORIZON):
        future_ts = ts[origins+h]
        hours = np.asarray(future_ts.hour)
        weekdays = np.asarray(future_ts.dayofweek)
        X = np.column_stack(
            [buf[:, -lag] for lag in LAGS] +
            [buf[:, -24:].mean(axis=1), buf[:, -24:].std(axis=1),
             buf.mean(axis=1),
             np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24),
             np.sin(2*np.pi*weekdays/7), np.cos(2*np.pi*weekdays/7)] +
            ([] if cov is None else [cov[origins+h, j]
                                     for j in range(cov.shape[1])]))
        zpred = reg.predict(X)
        pred[:, h] = zpred*sd+mu
        buf = np.column_stack((buf[:, 1:], zpred))
    assert np.isfinite(pred).all()
    actual = np.stack([values[g:g+HORIZON] for g in origins])
    return pred, actual


def make_points(site, gs, pred, actual):
    rows = []
    for j, g in enumerate(gs):
        future = site.timestamp.iloc[g:g+HORIZON].reset_index(drop=True)
        issue = future.iloc[0]-pd.Timedelta(hours=1)
        for h in range(HORIZON):
            rows.append(dict(origin_index=g, inferred_issue_time=issue,
                             target_timestamp=future.iloc[h], horizon_step=h+1,
                             actual=actual[j, h], prediction=pred[j, h]))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("wind", "solar"))
    parser.add_argument("--zone", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selected_all = pd.read_csv(ORIGINS)
    selected_all = selected_all.loc[selected_all.eligible]
    if args.track:
        selected_all = selected_all.loc[selected_all.track.eq(args.track)]
    if args.zone is not None:
        selected_all = selected_all.loc[selected_all.zone.eq(args.zone)]
    assert not selected_all.empty
    started = datetime.now(timezone.utc).isoformat()
    records = []
    for dataset, selected in selected_all.groupby("dataset", sort=True):
        track, zone = selected.track.iloc[0], int(selected.zone.iloc[0])
        site = pd.read_csv(BASE / f"data/processed/gefcom2014_{track}_clean.csv",
                           parse_dates=["timestamp"])
        site = site.loc[site.zone.eq(zone)].sort_values("timestamp").reset_index(drop=True)
        gs = selected.origin_index.astype(int).tolist()
        assert len(site) >= max(gs)+HORIZON
        for weather in (False, True):
            arm = "weather" if weather else "calendar"
            for seed in range(5):
                t0 = time.monotonic()
                reg, z, cov, mu, sd = train(site, seed, weather)
                pred, actual = forecast(reg, site, gs, z, cov, mu, sd)
                points = make_points(site, gs, pred, actual)
                path = OUT / f"{dataset}_{arm}_seed{seed}_points.csv"
                points.to_csv(path, index=False)
                records.append(dict(dataset=dataset, arm=arm, seed=seed,
                                    origins=len(gs), best_iteration=reg.best_iteration_,
                                    seconds=round(time.monotonic()-t0, 2),
                                    file=path.name, sha256=sha(path)))
                print(f"{dataset} {arm} seed={seed}: {len(gs)} origins, "
                      f"{records[-1]['seconds']} s", flush=True)
    manifest = ["# LightGBM competition-aligned CPU manifest", "",
                f"- Started UTC: {started}",
                f"- Finished UTC: {datetime.now(timezone.utc).isoformat()}",
                f"- Python: {sys.version.split()[0]}; LightGBM: {lgb.__version__}",
                f"- Host: {platform.platform()}",
                f"- Source script SHA-256: `{sha(Path(__file__))}`",
                f"- Candidate origins SHA-256: `{sha(ORIGINS)}`",
                "- Both arms: target lags and future-known calendar; weather arm "
                "also uses target-hour NWP. Hyperparameters, training rows, and "
                "early stopping follow the archived LightGBM runner.",
                "- All test windows are common to both arms and Chronos-2. "
                "Issue time is inferred from the competition schedule; original "
                "`results.csv` untouched.", "",
                "| Dataset | Arm | Seed | Origins | Best iteration | CPU seconds | File | SHA-256 |",
                "|---|---|---:|---:|---:|---:|---|---|"]
    for r in records:
        manifest.append(f"| {r['dataset']} | {r['arm']} | {r['seed']} | "
                        f"{r['origins']} | {r['best_iteration']} | {r['seconds']} | "
                        f"`{r['file']}` | `{r['sha256']}` |")
    (OUT / "run_manifest.md").write_text("\n".join(manifest)+"\n")


if __name__ == "__main__":
    main()
