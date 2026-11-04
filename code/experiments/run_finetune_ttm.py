"""TSFM fine-tuning pilot (RQ1 微调组, CPU-feasible member): TTM full fine-tune.

Protocol docs/05: full fine-tune and LoRA each one setting, lr grid {1e-5, 5e-5, 1e-4},
early stop patience 5. TTM (~1M params, r2) is the only protocol TSFM trainable on
this machine, so it anchors the fine-tuning machinery; Chronos/TimesFM/Moirai
fine-tuning runs on the rented GPU later, reusing this script's structure.

Setting: univariate, 512-step context -> 24h horizon, standardized on train,
early stop on val MSE (patience 5, max 15 epochs), evaluated at step=24 origins
with the same metrics as the rest of the zoo. Pilot = one seed (0).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HORIZON, append_results, load_series, metrics, rolling_origins,
                    seasonal_naive_denom, split_indices)

BASE = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CTX = 512
LRS = [1e-5, 5e-5, 1e-4]
MAX_EPOCHS = 15
PATIENCE = 5
STRIDE = 2  # subsample training windows (pilot; protocol run uses stride 1 on GPU)


def windows(z, gs, tss, cov_needed=False):
    xs, ys = [], []
    for g, ts_g in zip(gs, tss):
        if np.isnan(z[g - CTX:g]).any() or np.isnan(z[g:g + HORIZON]).any():
            continue
        xs.append(z[g - CTX:g])
        ys.append(z[g:g + HORIZON])
    return (torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="solar")
    ap.add_argument("--zone", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(BASE / f"data/processed/gefcom2014_{args.track}_clean.csv",
                     parse_dates=["timestamp"])
    g = (df[df["zone"] == args.zone].set_index("timestamp")["power"]
         .sort_index().asfreq("h"))
    ts = g.index
    values = g.to_numpy(dtype=np.float64)
    n = len(values)
    train_end, test_start = split_indices(n)
    mu = np.nanmean(values[:train_end])
    sd = np.nanstd(values[:train_end]) + 1e-8
    z = (values - mu) / sd

    train_gs = list(range(CTX, train_end - HORIZON + 1, STRIDE))
    val_gs = [train_end + o for o in range(CTX, (test_start - train_end) - HORIZON + 1, HORIZON)]
    test_gs = [test_start + o for o in rolling_origins(n - test_start)
               if not np.isnan(values[test_start + o:test_start + o + HORIZON]).any()]
    X_tr, y_tr = windows(z, train_gs, [ts[i] for i in train_gs])
    X_va, y_va = windows(z, val_gs, [ts[i] for i in val_gs])
    print(f"windows train={len(X_tr)} val={len(X_va)} test={len(test_gs)}")
    X_tr, y_tr, X_va, y_va = X_tr.to(DEVICE), y_tr.to(DEVICE), X_va.to(DEVICE), y_va.to(DEVICE)

    from tsfm_public import TinyTimeMixerForPrediction
    denom = seasonal_naive_denom(values, test_start)
    rows = []
    for lr in LRS:
        model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM", revision="main").to(DEVICE)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_val, best_state, patience = np.inf, None, 0
        t0 = time.time()
        diverged = False
        for epoch in range(MAX_EPOCHS):
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 256):
                idx = perm[i:i + 256]
                out = model(past_values=X_tr[idx]).prediction_outputs
                loss = torch.nn.functional.mse_loss(out[:, :HORIZON, :], y_tr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(
                    model(past_values=X_va).prediction_outputs[:, :HORIZON, :], y_va).item()
            if not np.isfinite(v):
                diverged = True
                print(f"  lr={lr}: non-finite val at epoch {epoch}")
                break
            if v < best_val - 1e-6:
                best_val, patience = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                patience += 1
                if patience >= PATIENCE:
                    break
        wall = time.time() - t0
        if best_state is not None and not diverged:
            model.load_state_dict(best_state)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(test_gs), 64):
                chunk = test_gs[i:i + 64]
                xb = torch.tensor(np.stack([z[t - CTX:t] for t in chunk]),
                                  dtype=torch.float32).unsqueeze(-1).to(DEVICE)
                out = model(past_values=xb).prediction_outputs[:, :HORIZON, 0].detach().cpu().numpy()
                preds.append(out)
        P = np.concatenate(preds) * sd + mu
        y_t = np.stack([values[t:t + HORIZON] for t in test_gs])
        m = metrics(P, y_t, denom)
        rows.append({**m, "model": f"ttm-ft", "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
                     "horizon": HORIZON, "seed": args.seed, "time_s": round(wall, 1),
                     "note": f"full finetune; lr={lr}; epochs={epoch + 1}; stride={STRIDE}"
                             + ("; DIVERGED" if diverged else ""),
                     "wall_s": round(wall, 1), "date": "2026-10-30"})
        print(f"lr={lr}: best_val={best_val:.5f} MASE={m['MASE_daylight']:.4f} wall={wall:.0f}s")

    append_results(rows)


if __name__ == "__main__":
    main()
