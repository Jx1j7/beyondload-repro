"""RQ6 routing pilot: volatility-tier routing between TSFM and specialized models.

Per docs/05 §2.6 (pilot, step=24 origins, solar/wind zone 1):
  - volatility metric: std of first differences over the last 24 steps of the
    input window (known at forecast time)
  - tiers: low/mid/high by train-period 33%/67% quantiles of that metric
  - rule: per-tier best model by validation MAE; test hour routed accordingly
  - controls: best single model, simple average ensemble, val-error-weighted ensemble
  - report: overall MAE (+MASE-like ratio vs seasonal naive) and mean per-origin
    inference cost (measured latency x routing share)

Candidates: chronos (TSFM), ttm (TSFM), lightgbm+cov (specialized+future NWP).
Output: experiments/routing_pilot.csv (append-only).
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


def chrono_preds(values, gs):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu")
    preds, lats = [], []
    for g in gs:
        ctx = torch.tensor(values[g - LOOKBACK:g], dtype=torch.float32)
        t0 = time.time()
        out = pipe.predict(inputs=ctx, prediction_length=HORIZON)
        lats.append(time.time() - t0)
        preds.append(out[0].median(dim=0).values.numpy())
    return np.array(preds), float(np.median(lats))


def ttm_preds(values, gs):
    from tsfm_public import TinyTimeMixerForPrediction
    model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM", revision="main")
    model.eval()
    preds, lats = [], []
    for g in gs:
        ctx = torch.tensor(values[g - 512:g], dtype=torch.float32).reshape(1, -1, 1)
        t0 = time.time()
        with torch.no_grad():
            out = model(past_values=ctx).prediction_outputs
        lats.append(time.time() - t0)
        preds.append(out[0, :HORIZON, 0].numpy())
    return np.array(preds), float(np.median(lats))


def lgbm_both(values, ts, test_start, step, use_cov=True):
    from run_trained import run_lightgbm
    from common import load_covariates
    cov = load_covariates(_TRACK, 1) if use_cov else None
    pred_test, _act, t_train, _it, _s, pred_val = run_lightgbm(
        values, cov, ts, test_start, seed=0, step=step, return_val=True)
    n_test = len(pred_test)
    return {"val": pred_val, "test": pred_test}, t_train / (n_test + len(pred_val))


def main():
    global _TRACK
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["solar", "wind"])
    ap.add_argument("--zone", type=int, default=1)
    ap.add_argument("--step", type=int, default=HORIZON)
    ap.add_argument("--no-nwp", action="store_true",
                    help="route between TSFMs and NWP-less LightGBM (deployment without NWP)")
    args = ap.parse_args()
    _TRACK = args.track

    ts, values = load_series(args.track, args.zone)
    n = len(values)
    train_end, test_start = split_indices(n)
    step = args.step

    # volatility metric per origin, computed identically for train/val/test tiers
    def vol_at(g):
        return float(np.std(np.diff(values[g - 24:g])))

    all_origins = list(range(LOOKBACK, n - HORIZON + 1, step))
    vols = np.array([vol_at(g) for g in all_origins])
    t1, t2 = np.quantile(vols[:train_end], [1 / 3, 2 / 3])
    tiers_all = np.digitize(vols, [t1, t2])  # 0 low, 1 mid, 2 high

    is_test = np.array([g >= test_start + LOOKBACK for g in all_origins])  # match model grids

    # ONE grid definition shared by all candidates:
    #  - test: rolling_origins convention, NaN-target origins dropped up front
    #  - val: run_lightgbm's own convention (train_end + local LOOKBACK offset)
    test_gs = [test_start + o for o in rolling_origins(n - test_start, step)
               if not np.isnan(values[test_start + o:test_start + o + HORIZON]).any()]
    val_gs = [train_end + o for o in range(LOOKBACK, (test_start - train_end) - HORIZON + 1, step)]
    tiers_val = np.digitize([vol_at(g) for g in val_gs], [t1, t2])
    tiers_test = np.digitize([vol_at(g) for g in test_gs], [t1, t2])
    y_val = np.stack([values[g:g + HORIZON] for g in val_gs])
    y_test = np.stack([values[g:g + HORIZON] for g in test_gs])

    candidates = {}
    p, lat_ch = chrono_preds(values, test_gs)
    pv, _ = chrono_preds(values, val_gs)
    candidates["chronos"] = {"val": pv, "test": p}
    p, lat_tt = ttm_preds(values, test_gs)
    pv, _ = ttm_preds(values, val_gs)
    candidates["ttm"] = {"val": pv, "test": p}
    candidates["lightgbm+cov"], lat_lg = lgbm_both(values, ts, test_start, step,
                                                   use_cov=not args.no_nwp)
    lgbm_name = "lightgbm+cov" if not args.no_nwp else "lightgbm"
    candidates[lgbm_name] = candidates.pop("lightgbm+cov")
    assert len(candidates[lgbm_name]["test"]) == len(test_gs), \
        f"{len(candidates[lgbm_name]['test'])} vs {len(test_gs)}"
    assert len(candidates[lgbm_name]["val"]) == len(val_gs)
    lats = {"chronos": lat_ch, "ttm": lat_tt, lgbm_name: lat_lg}

    names = list(candidates)
    val_mae = {}
    for nm in names:
        val_mae[nm] = {}
        for t in range(3):
            m = tiers_val == t
            if m.sum():
                val_mae[nm][t] = float(np.abs(candidates[nm]["val"][m] - y_val[m]).mean())

    # route test hours by tier -> best val model per tier
    tier_best = {}
    for t in range(3):
        tier_best[t] = min((d[t], nm) for nm, d in val_mae.items() if t in d)[1]
    routed = np.zeros_like(y_test)
    shares = {nm: 0.0 for nm in names}
    for t in range(3):
        m = tiers_test == t
        routed[m] = candidates[tier_best[t]]["test"][m]
        shares[tier_best[t]] += float(m.mean())

    ens_mean = np.mean([candidates[nm]["test"] for nm in names], axis=0)
    w = np.array([1.0 / max(val_mae[nm].get(0, 1), 1e-6) +
                  1.0 / max(val_mae[nm].get(1, 1), 1e-6) +
                  1.0 / max(val_mae[nm].get(2, 1), 1e-6) for nm in names])
    w = w / w.sum()
    ens_weighted = sum(wt * candidates[nm]["test"] for wt, nm in zip(w, names))

    naive_den = float(np.mean(np.abs(np.stack([values[g - HORIZON:g] for g in test_gs]) - y_test)))

    def mase(p):
        return float(np.abs(p - y_test).mean()) / naive_den

    rows = []
    def add(tag, p, cost=None, note=""):
        rows.append({"setup": tag, "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
                     "MAE_test": round(float(np.abs(p - y_test).mean()), 4),
                     "MASE_test": round(mase(p), 4),
                     "cost_ms_per_origin": None if cost is None else round(cost * 1000, 2),
                     "note": note})

    for nm in names:
        add(nm, candidates[nm]["test"], lats[nm], "single model")
    add("routed", routed, sum(shares[nm] * lats[nm] for nm in names),
        "tiers=" + ",".join(f"{t}:{tier_best[t]}" for t in range(3)))
    add("ens-mean", ens_mean, sum(lats[nm] for nm in names), "simple average")
    add("ens-weighted", ens_weighted, sum(lats[nm] for nm in names),
        "weights=" + ",".join(f"{nm}:{wt:.2f}" for nm, wt in zip(names, w)))

    out = pd.DataFrame(rows)
    csv = BASE / "experiments" / "routing_pilot.csv"
    out.to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(out.to_string(index=False))
    print(f"appended -> {csv}")


if __name__ == "__main__":
    main()
