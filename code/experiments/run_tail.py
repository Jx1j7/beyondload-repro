"""RQ5 tail pilot: ramp events and high-volatility hours on GEFCom2014.

Per docs/05 §2.4 (pilot variant, step=24 origins):
  - ramp event: 1h change > threshold (wind 0.20 / solar 0.30 of capacity; power is [0,1])
  - ramp-hours vs non-ramp-hours MAE ratio per model
  - ramp detection F1: predicted series shows the same threshold crossing
  - high-volatility quantile: top 10% test hours by 24h rolling std of actuals;
    MAE there, plus CRPS (quantile approximation) for chronos-bolt only.

Predictions are assembled into one contiguous test-span series (origins every 24h
tile the span), so diffs at window borders use the neighbouring window's values.
Output: experiments/tail_pilot.csv (append-only).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HORIZON, LOOKBACK, load_covariates, load_series, rolling_origins, split_indices)

BASE = Path(__file__).resolve().parents[2]
RAMP_TH = {"wind": 0.20, "solar": 0.30}
HIGHVOL_FRAC = 0.10
CHRONOS_QUANTILES = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def chrono_predictions(values, test_start, step=HORIZON):
    """Median + quantile forecasts per origin from Chronos-Bolt-Base."""
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu")
    med, q, lats = [], [], []
    for o in rolling_origins(len(values) - test_start, step):
        g = test_start + o
        ctx = torch.tensor(values[g - LOOKBACK:g], dtype=torch.float32)
        t0 = time.time()
        out = pipe.predict(inputs=ctx, prediction_length=HORIZON)  # [1, 9, H]
        lats.append(time.time() - t0)
        q.append(out[0].numpy())
        med.append(out[0].median(dim=0).values.numpy())
    return np.array(med), np.array(q), float(np.median(lats))


def naive_predictions(values, test_start, step=HORIZON):
    preds = [values[test_start + o - HORIZON:test_start + o]
             for o in rolling_origins(len(values) - test_start, step)]
    return np.array(preds)


def lightgbm_predictions(values, ts, test_start, step=HORIZON):
    from run_trained import run_lightgbm
    from common import load_covariates
    cov = load_covariates("solar" if "solar" in _TRACK else _TRACK, 1)
    pred, act, t_train, iters, skipped, _v = run_lightgbm(
        values, cov, ts, test_start, seed=0, step=step)
    return pred, act


def crps_from_quantiles(q: np.ndarray, y: np.ndarray) -> float:
    """CRPS approximated from K quantile forecasts: 2 * mean pinball loss."""
    taus = CHRONOS_QUANTILES.numpy()
    qs = q.reshape(-1, len(taus))
    ys = np.repeat(np.asarray(y).reshape(-1, 1), len(taus), axis=1)
    diff = ys - qs
    pinball = np.maximum(taus * diff, (taus - 1) * diff)
    return float(2.0 * pinball.mean())


def main():
    global _TRACK
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["solar", "wind"])
    ap.add_argument("--zone", type=int, default=1)
    ap.add_argument("--step", type=int, default=HORIZON)
    args = ap.parse_args()
    _TRACK = args.track

    ts, values = load_series(args.track, args.zone)
    n = len(values)
    _, test_start = split_indices(n)
    th = RAMP_TH[args.track]

    origins = [o for o in rolling_origins(n - test_start, args.step)
               if not np.isnan(values[test_start + o:test_start + o + HORIZON]).any()]
    g_vals = [test_start + o for o in origins]
    origin_set = set(origins)
    keep = [i for i, o in enumerate(rolling_origins(n - test_start, args.step))
            if o in origin_set]

    models = {}
    models["SeasonalNaive24"] = (naive_predictions(values, test_start, args.step)[keep], None, 0.0)
    med, qs, lat = chrono_predictions(values, test_start, args.step)
    med, qs = med[keep], qs[keep]
    models["chronos"] = (med, qs, lat)
    from run_trained import run_lightgbm
    cov = load_covariates(args.track, args.zone)
    pred, _act, _t, _it, _s, _v = run_lightgbm(values, cov, ts, test_start, seed=0, step=args.step)
    models["lightgbm+cov"] = (pred, None, None)

    # per-origin assembly: dropping a middle origin breaks contiguity, so every
    # series (actuals, previous-hour actuals, hour indices) is tiled per segment
    y = np.concatenate([values[g:g + HORIZON] for g in g_vals])
    yprev = np.concatenate([values[g - 1:g + HORIZON - 1] for g in g_vals])
    hours = np.concatenate([np.arange(g, g + HORIZON) for g in g_vals])
    A = np.abs(y - yprev) > th
    roll_std = pd.Series(values[test_start + origins[0]:], index=pd.date_range(
        ts[test_start + origins[0]], periods=n - test_start - origins[0], freq="h")).rolling(24).std().to_numpy()
    vol_th = np.nanquantile(roll_std, 1 - HIGHVOL_FRAC)
    HV = np.nan_to_num(roll_std[hours - (test_start + origins[0])], nan=-1) >= vol_th

    rows = []
    for name, (preds, qs_m, lat) in models.items():
        preds = np.asarray(preds)
        assert preds.shape[0] == len(g_vals), f"{name}: {preds.shape[0]} vs {len(g_vals)}"
        P = preds.reshape(-1)
        # border diffs use the actual previous hour of each segment
        P_prev = np.concatenate([
            np.concatenate([[values[g - 1]], p[:-1]]) for p, g in zip(preds, g_vals)])
        R = np.abs(P - P_prev) > th
        tp = float((R & A).sum()); fp = float((R & ~A).sum()); fn = float((~R & A).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
        mae_ramp = float(np.abs(P - y)[A].mean())
        mae_flat = float(np.abs(P - y)[~A].mean())
        row = {"model": name, "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
               "ramp_share": round(float(A.mean()), 4),
               "ramp_MAE": round(mae_ramp, 4), "flat_MAE": round(mae_flat, 4),
               "ramp_ratio": round(mae_ramp / mae_flat, 3),
               "ramp_F1": round(f1, 3),
               "highvol_MAE": round(float(np.abs(P - y)[HV].mean()), 4),
               "crps_highvol": None, "crps_all": None, "lat_ms": None}
        if qs_m is not None:
            qh = qs_m.transpose(0, 2, 1).reshape(-1, qs_m.shape[1])  # (n*H, K) aligned with y
            row["crps_highvol"] = round(crps_from_quantiles(qh[HV], y[HV]), 4)
            row["crps_all"] = round(crps_from_quantiles(qh, y), 4)
            row["lat_ms"] = round(lat * 1000, 1)
        rows.append(row)

    out = pd.DataFrame(rows)
    csv = BASE / "experiments" / "tail_pilot.csv"
    out.to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(out.to_string(index=False))
    print(f"appended -> {csv}")


if __name__ == "__main__":
    main()
