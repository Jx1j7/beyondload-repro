"""Shared utilities for beyondload experiments (data, splits, metrics, results I/O)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
HORIZON = 24
LOOKBACK = 168
DAYLIGHT_THRESHOLD = 0.01  # fraction of installed capacity (power is pre-normalized)
SEEDS = [0, 1, 2, 3, 4]


def load_series(track: str, zone: int):
    """Return (timestamps, values) for one zone of a cleaned GEFCom2014 track."""
    value_col = "load" if track == "load" else "power"
    df = pd.read_csv(BASE / f"data/processed/gefcom2014_{track}_clean.csv",
                     parse_dates=["timestamp"])
    g = (df[df["zone"] == zone].set_index("timestamp")[value_col]
         .sort_index().asfreq("h"))
    return g.index, g.to_numpy(dtype=np.float64)


def load_covariates(track: str, zone: int):
    """NWP covariate panel aligned with load_series (numeric cols except target)."""
    value_col = "load" if track == "load" else "power"
    df = pd.read_csv(BASE / f"data/processed/gefcom2014_{track}_clean.csv",
                     parse_dates=["timestamp"])
    g = df[df["zone"] == zone].set_index("timestamp").sort_index().asfreq("h")
    cov = g.drop(columns=[value_col]).select_dtypes("number")
    return cov


def split_indices(n: int, train=0.7, val=0.1):
    """Exact protocol split: train [0, train_end), val [train_end, val_end), test from val_end."""
    train_end = int(n * train)
    val_end = int(n * (train + val))
    return train_end, val_end


def rolling_origins(n_test: int, step: int = HORIZON):
    """Yield local origin o (context ends at global test_start + o).

    Context = series[o-LOOKBACK:o], target = series[o:o+HORIZON] (global indices
    after adding test_start). Skipped-NaN handling is left to callers.
    """
    return range(LOOKBACK, n_test - HORIZON + 1, step)


def collect_targets(values: np.ndarray, test_start: int, step: int = HORIZON):
    """Concatenated actuals over origins whose 24-step target is fully observed,
    plus the number of skipped origins (protocol >3h-drop rule)."""
    acts, skipped = [], 0
    for o in rolling_origins(len(values) - test_start, step):
        g = test_start + o
        tgt = values[g:g + HORIZON]
        if np.isnan(tgt).any():
            skipped += 1
            continue
        acts.append(tgt)
    return np.concatenate(acts), skipped


def metrics(pred: np.ndarray, act: np.ndarray, denom_mase: float) -> dict:
    """Both arrays shaped (n_origins, HORIZON); daylight mask restricts to observed power."""
    pred = np.asarray(pred).reshape(-1, HORIZON)
    act = np.asarray(act).reshape(-1, HORIZON)
    assert pred.shape == act.shape
    mask = act > DAYLIGHT_THRESHOLD
    mae = np.mean(np.abs(pred - act))
    rmse = np.sqrt(np.mean((pred - act) ** 2))
    mae_dl = np.mean(np.abs(pred[mask] - act[mask]))
    rmse_dl = np.sqrt(np.mean((pred[mask] - act[mask]) ** 2))
    return dict(MAE=mae, RMSE=rmse, nMAE=mae,  # capacity-normalized -> identical
                MAE_daylight=mae_dl, RMSE_daylight=rmse_dl,
                MASE_daylight=mae_dl / denom_mase)


RESULT_COLUMNS = ["model", "MAE", "RMSE", "nMAE", "MAE_daylight", "RMSE_daylight",
                  "MASE_daylight", "time_s", "dataset", "horizon", "seed", "date", "note", "wall_s"]


def append_results(rows: list[dict], csv_name: str = "results.csv") -> None:
    out = pd.DataFrame(rows)
    for c in out.columns:
        if out[c].dtype == object:  # notes must not contain commas (append-only CSV)
            out[c] = out[c].astype(str).str.replace(",", ";", regex=False)
    out = out.reindex(columns=RESULT_COLUMNS)  # fixed order regardless of dict order
    csv = BASE / "experiments" / csv_name
    out.to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"appended -> {csv}")


def seasonal_naive_denom(values: np.ndarray, test_start: int, step: int = HORIZON) -> float:
    """MASE denominator: daylight MAE of seasonal naive on the test set.
    NaN-target windows are skipped (protocol >3h-drop rule) — a single NaN
    window would otherwise poison the denominator for the whole dataset
    (bug found on wind zone3, fixed 2026-11-04)."""
    act, pred = [], []
    for o in rolling_origins(len(values) - test_start, step):
        g = test_start + o
        a = values[g:g + HORIZON]
        p = values[g - HORIZON:g]
        if np.isnan(a).any() or np.isnan(p).any():
            continue
        act.append(a)
        pred.append(p)
    act = np.concatenate(act)
    pred = np.concatenate(pred)
    mask = act > DAYLIGHT_THRESHOLD
    return float(np.mean(np.abs(pred[mask] - act[mask])))
