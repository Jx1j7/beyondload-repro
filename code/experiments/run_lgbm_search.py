"""LightGBM hyperparameter random search (protocol §2.3: 20 trials, shared space).

Upgrades the pilot recipe (fixed num_leaves=31, lr=0.05) to a searched recipe.
Selection metric: val MAE on daylight hours for solar, all hours otherwise
(matches the headline MASE_daylight convention). Test protocol unchanged
(step=24 origins, recursive 24-step rollout). Appends rows with model tag
"lightgbm+cov+srch" so the fixed-recipe pilot rows stay comparable.

Usage: .venv/bin/python code/experiments/run_lgbm_search.py --track solar [--trials 20]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HORIZON, append_results, load_covariates, load_series, metrics,
                    seasonal_naive_denom, split_indices)

BASE = Path(__file__).resolve().parents[2]
LAGS = [1, 2, 3, 24, 25, 48, 72, 96, 120, 144, 168]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["solar", "wind", "load"])
    ap.add_argument("--zone", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    import lightgbm as lgb
    rng = np.random.default_rng(args.seed)

    ts, values = load_series(args.track, args.zone)
    cov = load_covariates(args.track, args.zone)
    n = len(values)
    train_end, test_start = split_indices(n)
    mu = np.nanmean(values[:train_end])
    sd = np.nanstd(values[:train_end]) + 1e-8
    z = (values - mu) / sd
    zc, _, _ = None, None, None
    if cov is not None:
        cmu = cov.iloc[:train_end].mean()
        csd = cov.iloc[:train_end].std().replace(0, 1.0) + 1e-8
        zc = ((cov - cmu) / csd).to_numpy()

    def features(buf, ts_now, cov_now):
        v = np.asarray(buf)
        f = [v[-l] for l in LAGS]
        f += [v[-24:].mean(), v[-24:].std(), v[-168:].mean()]
        f += [np.sin(2 * np.pi * ts_now.hour / 24), np.cos(2 * np.pi * ts_now.hour / 24),
              np.sin(2 * np.pi * ts_now.dayofweek / 7), np.cos(2 * np.pi * ts_now.dayofweek / 7)]
        if cov_now is not None:
            f += list(np.asarray(cov_now))
        return f

    def row_targets(i0, i1):
        X, y = [], []
        for i in range(max(168, i0), i1):
            if np.isnan(z[i]) or not np.isfinite(z[i - 168:i]).all():
                continue
            X.append(features(z[i - 168:i], ts[i], None if zc is None else zc[i]))
            y.append(z[i])
        return np.array(X), np.array(y)

    X_tr, y_tr = row_targets(168, train_end)
    X_va, y_va = row_targets(train_end, test_start)

    SPACE = dict(num_leaves=(7, 127), learning_rate=(0.01, 0.15),
                 min_child_samples=(5, 60), feature_fraction=(0.5, 1.0),
                 bagging_fraction=(0.5, 1.0), reg_lambda=(0.0, 5.0))
    best = None
    t0 = time.time()
    for trial in range(args.trials):
        params = {k: (float(rng.uniform(*v)) if isinstance(v[0], float) else int(rng.integers(*v)))
                  for k, v in SPACE.items()}
        m = lgb.LGBMRegressor(
            n_estimators=1000, random_state=args.seed, verbose=-1, **params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        # selection metric: val residual MAE at the early-stopped iteration
        val_pred = m.predict(X_va, num_iteration=m.best_iteration_)
        sel = float(np.abs(val_pred - y_va).mean())
        if best is None or sel < best[0]:
            best = (sel, params, m.best_iteration_)
        print(f"  trial {trial + 1}: val_MAE={sel:.4f} best={best[0]:.4f} {params}")
    sel, params, iters = best
    t_search = time.time() - t0

    # retrain with best params (fit() already keeps best model; refit on train for clarity)
    reg = lgb.LGBMRegressor(n_estimators=1000, random_state=args.seed, verbose=-1, **params)
    reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)])

    def rollout(gs):
        preds, acts = [], []
        for g in gs:
            tgt = values[g:g + HORIZON]
            if np.isnan(tgt).any():
                continue
            buf = list(z[g - 168:g])
            out = []
            for h in range(HORIZON):
                ts_now = ts[g + h]
                cov_now = None if zc is None else zc[g + h]
                x = np.array(features(buf, ts_now, cov_now)).reshape(1, -1)
                p = float(reg.predict(x, num_iteration=reg.best_iteration_)[0])
                out.append(p)
                buf.append(p)
            preds.append(np.array(out) * sd + mu)
            acts.append(tgt)
        return np.array(preds), np.array(acts)

    test_gs = [test_start + o for o in range(168, n - test_start - HORIZON + 1, HORIZON)
               if not np.isnan(values[test_start + o:test_start + o + HORIZON]).any()]
    P, Y = rollout(test_gs)
    denom = seasonal_naive_denom(values, test_start)
    m = metrics(P, Y, denom)
    append_results([{**m, "model": "lightgbm+cov+srch",
                     "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
                     "horizon": HORIZON, "seed": args.seed,
                     "time_s": round(t_search, 1),
                     "note": (f"randsearch {args.trials} trials; iters={iters}; "
                              + "; ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}"
                                          for k, v in params.items())),
                     "wall_s": round(time.time() - t0, 1), "date": "2026-10-30"}])


if __name__ == "__main__":
    main()
