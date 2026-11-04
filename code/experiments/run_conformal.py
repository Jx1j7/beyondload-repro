"""Conformal pilot (RQ2 machinery): calibrate prediction intervals on val, report on test.

Two wrappers:
  1. Chronos-Bolt native quantiles -> CQR (conformalized quantile regression):
     per-horizon-step additive correction estimated from val residuals.
  2. LightGBM point forecast -> split conformal: per-horizon-step absolute
     residual quantile, symmetric intervals.

Metrics (all hours, nominal 90%): PICP, PINAW, Winkler score, mean interval
width. Val origins roll within [train_end, test_start) at --step grid; test
origins follow the same protocol as other runners.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HORIZON, LOOKBACK, append_results, load_series, rolling_origins,
                    split_indices)

BASE = Path(__file__).resolve().parents[2]
NOMINAL = 0.9
ALPHA = 0.1


def conformal_level(n: int) -> float:
    """Standard split-conformal quantile level ceil((n+1)(1-alpha))/n."""
    return float(np.ceil((n + 1) * NOMINAL) / n)


def cqr_scores(val_hi, val_lo, val_y):
    lvl = conformal_level(len(val_y))
    s_hi = np.quantile(val_hi - val_y, lvl, axis=0)
    s_lo = np.quantile(val_y - val_lo, lvl, axis=0)
    return s_hi, s_lo


def split_conformal_q(val_res_abs):
    return np.quantile(val_res_abs, conformal_level(len(val_res_abs)), axis=0)


def pi_metrics(lo, hi, y, tag):
    picp = float(np.mean((y >= lo) & (y <= hi)))
    pinaw = float(np.mean(hi - lo))
    # Winkler score for 90% interval, penalty 1/(alpha/2) each side... standard: (hi-lo) +
    # (2/alpha)*(lo-y if y<lo) + (2/alpha)*(y-hi if y>hi), alpha=0.1
    alpha = ALPHA
    w = (hi - lo) + np.where(y < lo, (2 / alpha) * (lo - y), 0) + np.where(y > hi, (2 / alpha) * (y - hi), 0)
    return {"model": tag, "PICP": picp, "PINAW": pinaw, "Winkler": float(np.mean(w)),
            "width_median": float(np.median(hi - lo))}


def chronos_quantiles(values, origins_g):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu")
    quant = torch.stack([
        pipe.predict(inputs=torch.tensor(values[g - LOOKBACK:g], dtype=torch.float32),
                     prediction_length=HORIZON)[0]  # (Q, H)
        for g in origins_g])
    return quant.numpy()  # (n_origins, Q, H); chronos quantiles: 9 levels 0.1..0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["chronos", "lightgbm"])
    ap.add_argument("--track", default="solar")
    ap.add_argument("--zone", type=int, default=1)
    ap.add_argument("--step", type=int, default=HORIZON)
    args = ap.parse_args()

    ts, values = load_series(args.track, args.zone)
    n = len(values)
    train_end, test_start = split_indices(n)

    def grid(lo, hi):
        return [lo + o for o in range(LOOKBACK, (hi - lo) - HORIZON + 1, args.step)]

    val_gs = grid(train_end, test_start)
    test_gs = grid(test_start, n)
    print(f"val origins={len(val_gs)} test origins={len(test_gs)}")

    if args.model == "chronos":
        t0 = time.time()
        qv = chronos_quantiles(values, val_gs)   # (n_v, 9, 24)
        qt = chronos_quantiles(values, test_gs)  # (n_t, 9, 24)
        # quantile levels for chronos-bolt: 0.1..0.9 equally spaced -> 0.05/0.95 unavailable;
        # use innermost available (0.1/0.9) as the ~80% native interval and recalibrate
        hi_v, lo_v = qv[:, -1, :], qv[:, 0, :]
        y_v = np.concatenate([values[g:g + HORIZON] for g in val_gs]).reshape(len(val_gs), HORIZON)
        y_t = np.concatenate([values[g:g + HORIZON] for g in test_gs]).reshape(len(test_gs), HORIZON)
        s_hi, s_lo = cqr_scores(hi_v, lo_v, y_v)
        hi_t, lo_t = qt[:, -1, :] + s_hi, qt[:, 0, :] - s_lo
        native = pi_metrics(qt[:, 0, :], qt[:, -1, :], y_t, "chronos-native80")
        clip = (0.0, 1.0) if args.track == "solar" else (None, None)  # only capacity-normalized tracks
        conf = pi_metrics(np.clip(lo_t, *clip) if clip[0] is not None else lo_t,
                          np.clip(hi_t, *clip) if clip[0] is not None else hi_t,
                          y_t, "chronos-cqr90")
        rows = [{**native, "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
                 "horizon": HORIZON, "note": "conformal pilot;nominal=0.8 native", "date": "2026-10-29"},
                {**conf, "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
                 "horizon": HORIZON, "note": "conformal pilot;cqr to nominal 0.9", "date": "2026-10-29"}]
        print(f"chronos quantile passes: {time.time()-t0:.0f}s")
    else:
        # reuse the exact run_trained recipe/model (no duplicate implementation)
        from run_trained import run_lightgbm
        from common import load_covariates
        cov = load_covariates(args.track, args.zone) if getattr(args, "covariates", False) else None
        pred_test, y_t_rows, _t, _it, _s, pred_val = run_lightgbm(
            values, cov, ts, test_start, seed=0, step=args.step, return_val=True)
        val_gs = grid(train_end, test_start)
        test_gs = grid(test_start, n)
        assert len(pred_val) == len(val_gs) and len(pred_test) == len(test_gs)
        y_v = np.concatenate([values[g:g + HORIZON] for g in val_gs]).reshape(len(val_gs), HORIZON)
        y_t = y_t_rows.reshape(len(test_gs), HORIZON)
        q = split_conformal_q(np.abs(pred_val - y_v))
        print("val resid p50/p90:", float(np.percentile(np.abs(pred_val - y_v), 50).round(4)),
              float(np.percentile(np.abs(pred_val - y_v), 90).round(4)))
        lo_t, hi_t = pred_test - q, pred_test + q
        clip = (0.0, 1.0) if args.track == "solar" else (None, None)
        conf = pi_metrics(np.clip(lo_t, *clip) if clip[0] is not None else lo_t,
                          np.clip(hi_t, *clip) if clip[0] is not None else hi_t,
                          y_t, "lightgbm-conformal90")
        rows = [{**conf, "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
                 "horizon": HORIZON, "note": "conformal pilot;split conformal nominal 0.9",
                 "date": "2026-10-29"}]
    import pandas as pd
    out = pd.DataFrame(rows)
    out = out.reindex(columns=["model", "PICP", "PINAW", "Winkler", "width_median",
                               "dataset", "horizon", "date", "note"])
    csv = Path(__file__).resolve().parents[2] / "experiments" / "conformal_pilot.csv"
    out.to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"appended -> {csv}")


if __name__ == "__main__":
    main()
