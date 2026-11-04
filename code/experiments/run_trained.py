"""From-scratch trained models on GEFCom2014 (protocol 168-step lookback, 70/10/20 split).

Usage: .venv/bin/python code/experiments/run_trained.py --model {lightgbm|patchtst|itransformer|timemixer|vmd_patchtst}
       [--track solar --zone 1] [--seed 0] [--step 24] [--covariates]

Pilot recipe (protocol version will add 20-trial random search):
  - TSLib models: standardized on train, Adam lr=1e-2, batch 256, early stop 5,
    max 30 epochs. With --covariates, NWP columns are extra input channels and
    only channel 0 (power) is scored; covariate stats use train split only.
  - LightGBM: lag/calendar (+NWP at target time when --covariates), recursive
    24-step rollout, early stop on val.
  - vmd_patchtst: per-context-window VMD into K modes as channels -> PatchTST,
    sum channel outputs (leakage-free: decomposition sees only the window).
  - Origins whose target contains NaN are skipped (protocol >3h-drop rule).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HORIZON, LOOKBACK, append_results, collect_targets, load_covariates,
                    load_series, metrics, rolling_origins, seasonal_naive_denom, split_indices)

BASE = Path(__file__).resolve().parents[2]
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"  # rented-GPU runs must use GPU
sys.path.insert(0, str(BASE / "code" / "tslib"))

TSLIB_MODULE_NAMES = {"dlinear": "DLinear", "patchtst": "PatchTST", "itransformer": "iTransformer", "timemixer": "TimeMixer"}

TSLIB_CONFIGS = {
    "dlinear": dict(moving_avg=25),
    "patchtst": dict(d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.1, factor=1, activation="gelu"),
    "itransformer": dict(d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.1, factor=1,
                         activation="gelu", embed="timeF", freq="h"),
    "timemixer": dict(d_model=64, e_layers=2, d_ff=128, dropout=0.1, embed="timeF", freq="h",
                      decomp_method="moving_avg", moving_avg=25, channel_independence=0,
                      down_sampling_layers=2, down_sampling_window=2, down_sampling_method="avg",
                      use_future_multi_scale_bias=False, use_norm=1, top_k=5, c_out=1, label_len=48),
    "vmd_patchtst": dict(d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.1, factor=1, activation="gelu"),
}

VMD_K = 3  # band-limited modes; a residual mode (w - sum(u)) is appended -> enc_in = VMD_K + 1


def vmd_decompose_window(window: np.ndarray) -> np.ndarray:
    """VMD of one 1-D window -> (VMD_K+1, T); last row is the residual so that
    sum over modes reconstructs the window exactly (vmdpy alone doesn't)."""
    from vmdpy import VMD
    u, _, _ = VMD(window, 2000, 0, VMD_K, 0, 1, 1e-7)  # alpha=2000, tau=0, K, DC=0, init=1, tol
    return np.vstack([u, window - u.sum(axis=0)])


def make_windows_multi(z: np.ndarray, cov: np.ndarray | None, lookback=LOOKBACK, horizon=HORIZON):
    """Windows for TSLib: x = [lookback, channels], y = [horizon, channels]."""
    xs, ys = [], []
    if cov is None:
        cov = np.zeros((len(z), 0))
    n_ch = 1 + cov.shape[1]
    block = np.column_stack([z, cov]) if cov.shape[1] else z[:, None]
    for i in range(lookback, len(z) - horizon + 1):
        xb, yb = block[i - lookback:i], block[i:i + horizon]
        if not np.isfinite(xb).all() or not np.isfinite(yb).all():
            continue  # drop windows touching NaN segments (>3h protocol drops)
        xs.append(xb)
        ys.append(yb)
    return (torch.tensor(np.array(xs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.float32))


def train_tslib(model, X_tr, y_tr, X_va, y_va, max_epochs=30, patience=5):
    model = model.to(DEVICE)
    X_tr, y_tr, X_va, y_va = X_tr.to(DEVICE), y_tr.to(DEVICE), X_va.to(DEVICE), y_va.to(DEVICE)
    best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    lr = 1e-2
    for attempt in range(2):  # retry once at lr/10 if optimization diverges
        model.load_state_dict(best_state)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_val, patience_ct = np.inf, 0
        t0 = time.time()
        diverged = False
        for epoch in range(max_epochs):
            model.train()
            for i in range(0, len(X_tr), 256):
                xb, yb = X_tr[i:i + 256], y_tr[i:i + 256]
                loss = torch.nn.functional.mse_loss(model(xb, None, None, None), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(model(X_va, None, None, None), y_va).item()
            if not np.isfinite(v):
                diverged = True
                break
            if v < best_val - 1e-6:
                best_val = v
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
                patience_ct = 0
            else:
                patience_ct += 1
                if patience_ct >= patience:
                    break
        if not diverged:
            break
        lr *= 0.1  # diverged: reset and retry at lower lr
        print(f"  diverged, retrying with lr={lr}")
    model.load_state_dict(best_state)
    model.eval()
    return time.time() - t0, epoch + 1


def standardize_cov(cov: pd.DataFrame, train_end: int):
    mu = cov.iloc[:train_end].mean()
    sd = cov.iloc[:train_end].std().replace(0, 1.0) + 1e-8
    return ((cov - mu) / sd).to_numpy(), mu.to_numpy(), sd.to_numpy()


def run_tslib(name, values, cov_values, test_start, seed, step):
    from argparse import Namespace
    import importlib
    torch.manual_seed(seed); np.random.seed(seed)
    train_end, _ = split_indices(len(values))
    train = values[:train_end]
    mu = np.nanmean(train)
    sd = np.nanstd(train) + 1e-8
    z = (values - mu) / sd
    zc = cov_values  # already standardized on train stats
    n_ch = 1 + (0 if zc is None else zc.shape[1])

    if name == "vmd_patchtst":
        return run_vmd_patchtst(z, values, train_end, test_start, seed, step)

    X_tr, y_tr = make_windows_multi(z[:train_end], None if zc is None else zc[:train_end])
    X_va, y_va = make_windows_multi(z[train_end:test_start], None if zc is None else zc[train_end:test_start])
    cfg = Namespace(task_name="long_term_forecast", seq_len=LOOKBACK, pred_len=HORIZON,
                    enc_in=n_ch, **TSLIB_CONFIGS[name])
    mod = importlib.import_module(f"models.{TSLIB_MODULE_NAMES[name]}")
    model = mod.Model(cfg)
    t_train, epochs = train_tslib(model, X_tr, y_tr, X_va, y_va)

    # batched inference over all valid origins (verified equivalent to per-origin)
    ctxs, tgts = [], []
    for o in rolling_origins(len(values) - test_start, step):
        g = test_start + o
        tgt = values[g:g + HORIZON]
        if np.isnan(tgt).any() or np.isnan(z[g - LOOKBACK:g]).any():
            continue
        block = np.column_stack([z[g - LOOKBACK:g], zc[g - LOOKBACK:g]]) if zc is not None else z[g - LOOKBACK:g, None]
        ctxs.append(block)
        tgts.append(tgt)
    preds = []
    with torch.no_grad():
        for i in range(0, len(ctxs), 256):
            xb = torch.tensor(np.stack(ctxs[i:i + 256]), dtype=torch.float32).to(DEVICE)
            out = model(xb, None, None, None)[:, :, 0].detach().cpu().numpy()
            preds.append(out * sd + mu)
    return np.concatenate(preds), np.concatenate(tgts), t_train, epochs, 0


VMD_TRAIN_STRIDE = 4  # pilot: subsample training windows (VMD is ~0.2s per window)


def run_vmd_patchtst(z, values, train_end, test_start, seed, step):
    """Consistent mode-space model: train and infer on per-window VMD channels."""
    from argparse import Namespace
    import importlib

    mu = values[:train_end].mean()
    sd = values[:train_end].std() + 1e-8

    def window_modes(w):
        modes = vmd_decompose_window(w)          # (K, T)
        m = modes.mean(axis=1, keepdims=True)
        s = modes.std(axis=1, keepdims=True) + 1e-8
        return (modes - m) / s, m[:, 0], s[:, 0]  # normalized modes + per-mode stats

    # training windows (target = normalized modes of the future window)
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for i in range(LOOKBACK, train_end - HORIZON + 1, VMD_TRAIN_STRIDE):
        xin, _, _ = window_modes(z[i - LOOKBACK:i])
        xtg, _, _ = window_modes(z[i:i + HORIZON])
        tr_x.append(xin.T); tr_y.append(xtg.T)
    va_start = train_end
    for i in range(va_start + LOOKBACK, test_start - HORIZON + 1, VMD_TRAIN_STRIDE):
        xin, _, _ = window_modes(z[i - LOOKBACK:i])
        xtg, _, _ = window_modes(z[i:i + HORIZON])
        va_x.append(xin.T); va_y.append(xtg.T)
    X_tr = torch.tensor(np.array(tr_x), dtype=torch.float32)
    y_tr = torch.tensor(np.array(tr_y), dtype=torch.float32)
    X_va = torch.tensor(np.array(va_x), dtype=torch.float32)
    y_va = torch.tensor(np.array(va_y), dtype=torch.float32)

    cfg = Namespace(task_name="long_term_forecast", seq_len=LOOKBACK, pred_len=HORIZON,
                    enc_in=VMD_K + 1, **TSLIB_CONFIGS["vmd_patchtst"])
    mod = importlib.import_module("models.PatchTST")
    model = mod.Model(cfg)
    t_train, epochs = train_tslib(model, X_tr, y_tr, X_va, y_va, max_epochs=10)

    preds, acts, skipped = [], [], 0
    with torch.no_grad():
        for o in rolling_origins(len(values) - test_start, step):
            g = test_start + o
            tgt = values[g:g + HORIZON]
            w = z[g - LOOKBACK:g]
            if np.isnan(tgt).any() or np.isnan(w).any():
                skipped += 1
                continue
            xin, wm, ws = window_modes(w)
            ctx = torch.tensor(xin.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            p = model(ctx, None, None, None)[0].detach().cpu().numpy()  # (H, K) normalized modes
            p = p.T * ws[:, None] + wm[:, None]          # de-norm per mode
            preds.append(p.sum(axis=0) * sd + mu)        # sum modes -> power
            acts.append(tgt)
    return np.concatenate(preds), np.concatenate(acts), t_train, epochs, skipped


LAGS = [1, 2, 3, 24, 25, 48, 72, 96, 120, 144, 168]


def run_lightgbm(values, cov, ts, test_start, seed, step, train_window=None, return_val=False):
    import lightgbm as lgb

    train_end, _ = split_indices(len(values))
    if train_window is not None:
        # supervised rows restricted to the last `train_window` hours of the train
        # period (lag features may still reference preceding context)
        row_start = max(LOOKBACK, train_end - train_window)
    else:
        row_start = LOOKBACK
    train = values[:train_end]
    val = values[train_end:test_start]
    mu = np.nanmean(train)
    sd = np.nanstd(train) + 1e-8
    z = (values - mu) / sd
    cov_mu = cov_mu_sd = None
    if cov is not None:
        zc, cov_mu, cov_sd = standardize_cov(cov, train_end)
    else:
        zc = None

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
        for i in range(max(LOOKBACK, i0), i1):
            if np.isnan(z[i]) or not np.isfinite(z[i - LOOKBACK:i]).all():
                continue  # skip NaN-target rows and rows whose lag window has NaN
            X.append(features(z[i - LOOKBACK:i], ts[i], None if zc is None else zc[i]))
            y.append(z[i])
        return np.array(X), np.array(y)

    X_tr, y_tr = row_targets(row_start, train_end)
    X_va, y_va = row_targets(train_end, test_start)
    reg = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                            subsample=0.9, colsample_bytree=0.9, random_state=seed, verbose=-1)
    t0 = time.time()
    reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    t_train = time.time() - t0

    def rollout(gs):
        preds, acts = [], []
        for g in gs:
            tgt = values[g:g + HORIZON]
            if np.isnan(tgt).any():
                continue
            buf = list(z[g - LOOKBACK:g])
            out = []
            for h in range(HORIZON):
                ts_now = ts[g + h]
                cov_now = None if zc is None else zc[g + h]  # NWP known in advance
                x = np.array(features(buf, ts_now, cov_now)).reshape(1, -1)
                p = float(reg.predict(x)[0])
                out.append(p)
                buf.append(p)  # recursive rollout
            preds.append(np.array(out) * sd + mu)
            acts.append(tgt)
        return np.array(preds), np.array(acts)

    test_gs = [test_start + o for o in rolling_origins(len(values) - test_start, step)]
    preds, acts = rollout(test_gs)
    val_preds = None
    if return_val:
        train_end_v, _ = split_indices(len(values))
        val_gs = [train_end_v + o for o in range(LOOKBACK, (test_start - train_end_v) - HORIZON + 1, step)]
        val_preds, _ = rollout(val_gs)
    return preds, acts, round(t_train, 1), reg.best_iteration_, 0, val_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["lightgbm", "dlinear", "patchtst", "itransformer", "timemixer", "vmd_patchtst"])
    ap.add_argument("--track", default="solar")
    ap.add_argument("--zone", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=HORIZON)
    ap.add_argument("--covariates", action="store_true")
    ap.add_argument("--train_window", type=int, default=None,
                    help="RQ3: restrict supervised rows to last N hours of train period")
    args = ap.parse_args()

    ts, values = load_series(args.track, args.zone)
    cov = load_covariates(args.track, args.zone) if args.covariates else None
    cov_values = None
    if cov is not None:
        train_end, _ = split_indices(len(values))
        cov_values, _, _ = standardize_cov(cov, train_end)
    n = len(values)
    _, test_start = split_indices(n)
    denom = seasonal_naive_denom(values, test_start, args.step)

    print(f"[run_trained] device={DEVICE}")
    t0 = time.time()
    if args.model == "lightgbm":
        pred, act, t_train, note_epochs, skipped, _val = run_lightgbm(
            values, cov, ts, test_start, args.seed, args.step,
            train_window=args.train_window)
    else:
        pred, act, t_train, note_epochs, skipped = run_tslib(args.model, values, cov_values, test_start, args.seed, args.step)
    m = metrics(pred, act, denom)
    cov_tag = "+cov" if args.covariates else ""
    append_results([{**m, "model": args.model + cov_tag,
                     "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
                     "horizon": HORIZON, "seed": args.seed, "time_s": t_train,
                     "note": (f"trained; iters={note_epochs}; step={args.step}; skipped={skipped}"
                              + (f"; train_window={args.train_window}" if args.train_window else "")),
                     "wall_s": round(time.time() - t0, 1), "date": "2026-10-29"}])


if __name__ == "__main__":
    main()
