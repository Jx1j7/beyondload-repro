"""Phase-0 smoke experiment: GEFCom2014 Solar zone 1, day-ahead (24h) forecasting.

Purpose: prove the whole pipeline end-to-end (data -> split -> models -> metrics
-> results file), NOT to produce protocol-grade numbers. Deviations from
docs/05: rolling origins every 24h (protocol says every 1h), DLinear trained
with a fixed minimal recipe instead of 20-trial random search, single seed.

Models:
  - SeasonalNaive(24)          (statsforecast)
  - DLinear                    (TSLib, trained on train split, standardized)
  - Chronos-Bolt-Base zero-shot(chronos-forecasting)

Split: chronological 70/10/20 per docs/05 §2.2. Metrics: MAE / RMSE / MASE
(vs seasonal naive) on all hours and on daylight hours only (observed power
> 1% capacity, per docs/05 §2.4 since power is already capacity-normalized).

Output: appends rows to experiments/results.csv and prints a table.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "code" / "tslib"))

HORIZON = 24
LOOKBACK = 168
SEED = 0
DAYLIGHT_THRESHOLD = 0.01  # fraction of installed capacity


def load_series() -> pd.Series:
    df = pd.read_csv(BASE / "data/processed/gefcom2014_solar_clean.csv",
                     parse_dates=["timestamp"])
    s = df[df["zone"] == 1].set_index("timestamp")["power"].sort_index()
    s = s.asfreq("h")
    assert s.isna().sum() == 0, "unexpected NaNs after cleaning"
    return s


def rolling_origins(test: np.ndarray, test_start_idx: int, step: int = HORIZON):
    """Yield (context_end_idx_in_full_series, horizon_slice) tuples."""
    n = len(test)
    origins = range(LOOKBACK, n - HORIZON + 1, step)
    for o in origins:
        # context = last LOOKBACK points before origin within test (or earlier data)
        ctx_end = test_start_idx + o
        yield ctx_end, slice(ctx_end, ctx_end + HORIZON)


def run_seasonal_naive(series: np.ndarray, test_start: int, step=HORIZON):
    preds, actuals, masks = [], [], []
    for ctx_end, hs in rolling_origins(series[test_start:], test_start, step):
        context = series[ctx_end - LOOKBACK:ctx_end]
        p = np.tile(context[-HORIZON:], 1)  # yesterday = last 24 points of context
        preds.append(p)
        actuals.append(series[hs])
        # daylight mask per protocol: observed power > 1% capacity
        m = series[hs] > DAYLIGHT_THRESHOLD
        masks.append(m)
    return np.concatenate(preds), np.concatenate(actuals), np.concatenate(masks)


def run_dlinear(series: np.ndarray, test_start: int, seed: int, step=HORIZON):
    from models import DLinear

    torch.manual_seed(seed)
    np.random.seed(seed)
    train = series[:int(test_start * 0.875)]  # train+val boundary: val is last 1/8 of pre-test
    val = series[int(test_start * 0.875):test_start]
    mu, sd = train.mean(), train.std() + 1e-8
    z = (series - mu) / sd

    def windows(a):
        xs, ys = [], []
        for i in range(LOOKBACK, len(a) - HORIZON + 1):
            xs.append(a[i - LOOKBACK:i])
            ys.append(a[i:i + HORIZON])
        return (torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1))

    X_tr, y_tr = windows(train)
    X_va, y_va = windows(val)

    cfg = argparse.Namespace(task_name="long_term_forecast", seq_len=LOOKBACK,
                             pred_len=HORIZON, moving_avg=25, enc_in=1)
    model = DLinear.Model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    best_val, best_state, patience = np.inf, None, 0
    t0 = time.time()
    for epoch in range(30):
        model.train()
        for i in range(0, len(X_tr), 256):
            xb, yb = X_tr[i:i + 256], y_tr[i:i + 256]
            loss = torch.nn.functional.mse_loss(model(xb, None, None, None), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = torch.nn.functional.mse_loss(model(X_va, None, None, None), y_va).item()
        if v < best_val - 1e-6:
            best_val, best_state, patience = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(best_state)
    train_time = time.time() - t0

    preds, actuals, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for ctx_end, hs in rolling_origins(series[test_start:], test_start, step):
            ctx = torch.tensor(z[ctx_end - LOOKBACK:ctx_end], dtype=torch.float32).reshape(1, LOOKBACK, 1)
            p = model(ctx, None, None, None).squeeze().numpy() * sd + mu
            preds.append(p)
            actuals.append(series[hs])
            masks.append(series[hs] > DAYLIGHT_THRESHOLD)
    return (np.concatenate(preds), np.concatenate(actuals),
            np.concatenate(masks), train_time, epoch + 1)


def run_chronos(series: np.ndarray, test_start: int, step=HORIZON):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu")
    preds, actuals, masks, median_lat = [], [], [], []
    for ctx_end, hs in rolling_origins(series[test_start:], test_start, step):
        ctx = torch.tensor(series[ctx_end - LOOKBACK:ctx_end], dtype=torch.float32)
        t0 = time.time()
        q = pipe.predict(inputs=ctx, prediction_length=HORIZON)  # [1, num_quantiles, 24]
        median_lat.append(time.time() - t0)
        p = q[0].median(dim=0).values.numpy()
        preds.append(p)
        actuals.append(series[hs])
        masks.append(series[hs] > DAYLIGHT_THRESHOLD)
    return np.concatenate(preds), np.concatenate(actuals), np.concatenate(masks), float(np.median(median_lat))


def metrics(pred: np.ndarray, act: np.ndarray, mask: np.ndarray, denom: float):
    def mae_rmse(p, a):
        return np.mean(np.abs(p - a)), np.sqrt(np.mean((p - a) ** 2))
    mae_all, rmse_all = mae_rmse(pred, act)
    mae_dl, rmse_dl = mae_rmse(pred[mask], act[mask])
    return dict(MAE=mae_all, RMSE=rmse_all, nMAE=mae_all,  # capacity = 1
                MAE_daylight=mae_dl, RMSE_daylight=rmse_dl,
                MASE_daylight=mae_dl / denom)


def main():
    series = load_series().to_numpy()
    n = len(series)
    train_end, val_end = int(n * 0.7), int(n * 0.8)
    test_start = val_end
    print(f"series: {n} hours, test starts at {test_start} "
          f"({pd.Timestamp('2012-04-01') + pd.Timedelta(hours=test_start)})")

    rows = []

    p_naive, a, m = run_seasonal_naive(series, test_start)
    denom = np.mean(np.abs(p_naive[m] - a[m]))
    rows.append(("SeasonalNaive24", *metrics(p_naive, a, m, denom).values(), 0.0))

    p_dl, a2, m2, t_train, ep = run_dlinear(series, test_start, SEED)
    assert np.array_equal(m, m2)
    rows.append(("DLinear", *metrics(p_dl, a2, m2, denom).values(), t_train))

    p_ch, a3, m3, lat = run_chronos(series, test_start)
    assert np.array_equal(m, m3)
    rows.append(("ChronosBoltBase-zs", *metrics(p_ch, a3, m3, denom).values(), lat))

    cols = ["model", "MAE", "RMSE", "nMAE", "MAE_daylight", "RMSE_daylight", "MASE_daylight", "time_s"]
    out = pd.DataFrame([r if isinstance(r[0], str) else r for r in rows], columns=cols)
    out["dataset"] = "gefcom2014_solar_zone1"
    out["horizon"] = HORIZON
    out["seed"] = SEED
    out["date"] = pd.Timestamp.today().date().isoformat()
    exp = BASE / "experiments"
    exp.mkdir(exist_ok=True)
    csv = exp / "results.csv"
    out.to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"appended -> {csv}")


if __name__ == "__main__":
    main()
