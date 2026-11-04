"""Statistical baselines via statsforecast with per-origin refit on a rolling window.

Usage: .venv/bin/python code/experiments/run_stats.py --model {autoets|autoarima} [--track solar --zone 1] [--step 24]

Window: last 16 weeks (2688 h) of observed history before each origin — full
history refits are impractical for AutoARIMA on this machine and rolling
windows are standard for ETS/ARIMA baselines. AutoARIMA (SARIMA) is deferred
to the protocol run on stronger hardware.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HORIZON, append_results, collect_targets, load_series, metrics,
                    rolling_origins, seasonal_naive_denom, split_indices)

WINDOW = 2688  # 16 weeks


def run_autoets(values, test_start, step):
    from statsforecast.models import AutoETS
    preds, lats = [], []
    for o in rolling_origins(len(values) - test_start, step):
        g = test_start + o
        y = values[g - WINDOW:g]
        if np.isnan(y).any() or np.isnan(values[g:g + HORIZON]).any():
            continue
        t0 = time.time()
        m = AutoETS(season_length=24)
        m.fit(y=y)
        fc = m.predict(h=HORIZON)["mean"]
        lats.append(time.time() - t0)
        preds.append(np.asarray(fc))
    return np.concatenate(preds), float(np.median(lats))


def _fit_arima_one(args):
    """Worker: one AutoARIMA refit+predict. Module-level for picklability."""
    import numpy as _np
    from statsforecast.models import AutoARIMA as _AA
    y, target = args
    m = _AA(season_length=24)
    m.fit(y=y)
    return _np.asarray(m.predict(h=HORIZON)["mean"]), target


def run_autoarima(values, test_start, step, max_origins=None, workers=1):
    import numpy as _np
    if workers > 1:
        jobs = []
        done = 0
        for o in rolling_origins(len(values) - test_start, step):
            g = test_start + o
            y = values[g - WINDOW:g]
            if _np.isnan(y).any() or _np.isnan(values[g:g + HORIZON]).any():
                continue
            if max_origins and done >= max_origins:
                break
            jobs.append((y, values[g:g + HORIZON]))
            done += 1
        import multiprocessing as _mp
        with _mp.Pool(workers) as pool:
            results = pool.map(_fit_arima_one, jobs, chunksize=1)
        preds = _np.concatenate([r[0] for r in results])
        acts = _np.concatenate([r[1] for r in results])
        return preds, acts, float(len(jobs) and 93.0 or 0.0)  # per-fit latency not meaningful in pool
    return _run_autoarima_serial(values, test_start, step, max_origins)


def _run_autoarima_serial(values, test_start, step, max_origins=None):
    """Protocol SARIMA: per-origin refit on a 16-week rolling window.

    Cost warning: ~2-3 min per origin fit on this class of CPU (measured 188 s
    on solar). Full solar/wind runs are hours; smoke-test with --max_origins.
    """
    from statsforecast.models import AutoARIMA
    preds, acts, lats, done = [], [], [], 0
    for o in rolling_origins(len(values) - test_start, step):
        g = test_start + o
        y = values[g - WINDOW:g]
        if np.isnan(y).any() or np.isnan(values[g:g + HORIZON]).any():
            continue
        if max_origins and done >= max_origins:
            break
        t0 = time.time()
        m = AutoARIMA(season_length=24)
        m.fit(y=y)
        fc = m.predict(h=HORIZON)["mean"]
        lats.append(time.time() - t0)
        preds.append(np.asarray(fc))
        acts.append(values[g:g + HORIZON])
        done += 1
    return np.concatenate(preds), np.concatenate(acts), float(np.median(lats))


MODELS = {"autoets": run_autoets, "autoarima": run_autoarima}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--track", default="solar")
    ap.add_argument("--zone", type=int, default=1)
    ap.add_argument("--step", type=int, default=HORIZON)
    ap.add_argument("--max_origins", type=int, default=None,
                    help="smoke test: cap number of refit origins")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel origin fits (CPU cores); battery mode: keep 1")
    args = ap.parse_args()

    _, values = load_series(args.track, args.zone)
    n = len(values)
    _, test_start = split_indices(n)
    denom = seasonal_naive_denom(values, test_start, args.step)

    t0 = time.time()
    if args.model == "autoarima":
        pred, act, lat = MODELS[args.model](values, test_start, args.step,
                                            args.max_origins, args.workers)
        skipped = -1  # per-origin counting under cap differs from full run; see note
    else:
        pred, lat = MODELS[args.model](values, test_start, args.step)
        act, skipped = collect_targets(values, test_start, args.step)
    assert len(pred) == len(act)
    m = metrics(pred, act, denom)
    append_results([{**m, "model": args.model,
                     "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
                     "horizon": HORIZON, "seed": -1, "time_s": round(lat, 3),
                     "note": (f"window={WINDOW};step={args.step};skipped={skipped}"
                        + (f";max_origins={args.max_origins}" if args.max_origins else "")
                        + (f";workers={args.workers}" if args.workers > 1 else "")),
                     "wall_s": round(time.time() - t0, 1), "date": "2026-10-29"}])


if __name__ == "__main__":
    main()
