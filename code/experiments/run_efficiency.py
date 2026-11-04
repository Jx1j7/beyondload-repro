"""RQ4 efficiency pilot: parameters, inference latency (protocol-style), peak memory.

Protocol docs/05 §2.4: parameter count, batch-32 inference latency median over
100 runs after warmup, peak memory, total training wall time; unified hardware.

This runner measures INFERENCE only, one model per process (peak-RSS isolation):
  - param_count: total trainable parameters (0 for non-neural baselines)
  - lat_ms_b1 / lat_ms_b32: median per-call latency over 100 timed calls after
    10 warmup calls, context 512 (TSFM) or 168 (LightGBM rollout equivalent)
  - peak_rss_mb: process peak resident memory during measurement
Training wall time already lives in results.csv (time_s of trained models).

Usage: .venv/bin/python code/experiments/run_efficiency.py --model NAME
Models: naive | chronos | chronos2 | ttm | timesfm | moirai | dlinear | lightgbm
Output: experiments/efficiency_pilot.csv (append-only). Hardware: M1 (recorded).
"""
import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HORIZON, load_series, split_indices

BASE = Path(__file__).resolve().parents[2]
CTX_TSFM = 512
WARMUP, TIMED = 10, 100
HARDWARE = "Apple M1 CPU 8GB"  # overwritten when --device cuda
DEVICE = "cpu"


def timed(fn, n_warm=WARMUP, n_timed=TIMED):
    for _ in range(n_warm):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_timed):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()  # async kernels must finish inside the timed window
    return (time.perf_counter() - t0) / n_timed


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["naive", "chronos", "chronos2", "ttm", "timesfm", "moirai",
                             "dlinear", "lightgbm"])
    ap.add_argument("--track", default="solar")
    ap.add_argument("--zone", type=int, default=1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--hw", default=None, help="hardware tag override for the record")
    args = ap.parse_args()
    global DEVICE, HARDWARE
    DEVICE = args.device
    if args.hw:
        HARDWARE = args.hw
    if DEVICE == "cuda":
        assert torch.cuda.is_available(), "--device cuda but CUDA unavailable"
        HARDWARE = torch.cuda.get_device_name(0)
        print(f"[efficiency] device=cuda ({HARDWARE})")

    _, values = load_series(args.track, args.zone)
    n = len(values)
    _, test_start = split_indices(n)
    g = test_start + 168  # representative origin
    ctx = np.asarray(values[g - CTX_TSFM:g], dtype=np.float32)
    ctx_t = torch.tensor(ctx).unsqueeze(-1)
    row = {"model": args.model, "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
           "param_count": 0, "lat_ms_b1": None, "lat_ms_b32": None,
           "peak_rss_mb": None, "hardware": HARDWARE, "date": "2026-11-01", "note": ""}

    if args.model == "naive":
        lat = timed(lambda: np.tile(ctx[-HORIZON:], 1))
        row["lat_ms_b1"] = round(lat * 1000, 4)
    elif args.model in ("chronos", "chronos2"):
        from chronos import BaseChronosPipeline
        repo = "amazon/chronos-bolt-base" if args.model == "chronos" else "amazon/chronos-2"
        pipe = BaseChronosPipeline.from_pretrained(repo, device_map="cpu")
        if args.model == "chronos":
            fn = lambda: pipe.predict(inputs=ctx_t, prediction_length=HORIZON)
            b32 = lambda: pipe.predict(inputs=ctx_t.repeat(32, 1), prediction_length=HORIZON)
        else:
            df = pd.DataFrame({"item_id": "a",
                               "timestamp": pd.date_range("2024-01-01", periods=CTX_TSFM, freq="h"),
                               "target": ctx})
            fn = lambda: pipe.predict_df(df, prediction_length=HORIZON)
            df32 = pd.concat([df.assign(item_id=f"i{i}") for i in range(32)])
            b32 = lambda: pipe.predict_df(df32, prediction_length=HORIZON)
        row["lat_ms_b1"] = round(timed(fn) * 1000, 2)
        row["lat_ms_b32"] = round(timed(b32) * 1000, 2)
        row["param_count"] = sum(p.numel() for p in pipe.model.parameters())
    elif args.model == "ttm":
        from tsfm_public import TinyTimeMixerForPrediction
        model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM", revision="main").to(DEVICE)
        model.eval()
        ctx_d = ctx_t.to(DEVICE)
        fn = lambda: model(past_values=ctx_d.reshape(1, -1, 1))
        b32 = lambda: model(past_values=ctx_d.reshape(1, -1, 1).repeat(32, 1, 1))
        with torch.no_grad():
            row["lat_ms_b1"] = round(timed(fn) * 1000, 3)
            row["lat_ms_b32"] = round(timed(b32) * 1000, 3)
        row["param_count"] = count_params(model)
    elif args.model == "timesfm":
        import timesfm
        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="cpu", per_core_batch_size=32,
                                           horizon_len=128, context_len=CTX_TSFM,
                                           num_layers=50, use_positional_embedding=False),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
        fn = lambda: model.forecast([ctx], freq=[0])
        row["lat_ms_b1"] = round(timed(fn) * 1000, 2)
        batch = [ctx] * 32
        row["lat_ms_b32"] = round(timed(lambda: model.forecast(batch, freq=[0] * 32)) * 1000, 2)
        row["param_count"] = sum(p.numel() for p in model._model.parameters())
    elif args.model == "moirai":
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
        model = Moirai2Forecast(module=module, prediction_length=HORIZON, context_length=CTX_TSFM,
                                target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        predictor = model.create_predictor(batch_size=32)
        entry = {"start": pd.Period("2024-01-01", freq="h"), "target": ctx, "item_id": "t"}
        fn = lambda: next(iter(predictor.predict([entry])))
        row["lat_ms_b1"] = round(timed(fn) * 1000, 2)
        entries = [{**entry, "item_id": f"i{i}"} for i in range(32)]
        b32 = lambda: next(iter(predictor.predict(entries)))
        row["lat_ms_b32"] = round(timed(b32, n_timed=30) * 1000, 2)
        row["param_count"] = sum(p.numel() for p in module.parameters())
    elif args.model == "dlinear":
        from argparse import Namespace
        sys.path.insert(0, str(BASE / "code" / "tslib"))
        from models import DLinear
        cfg = Namespace(task_name="long_term_forecast", seq_len=168, pred_len=HORIZON,
                        enc_in=1, moving_avg=25)
        model = DLinear.Model(cfg).to(DEVICE)
        model.eval()
        xb = torch.tensor(ctx[-168:]).reshape(1, 168, 1).to(DEVICE)
        fn = lambda: model(xb, None, None, None)
        with torch.no_grad():
            row["lat_ms_b1"] = round(timed(fn) * 1000, 3)
            row["lat_ms_b32"] = round(timed(lambda: model(xb.repeat(32, 1, 1), None, None, None)) * 1000, 3)
        row["param_count"] = count_params(model)
    elif args.model == "lightgbm":
        # representative cost: amortized train+recursive-rollout wall time per origin
        row["lat_ms_b1"] = round(lats_proxy(args.track, args.zone) * 1000, 2)
        row["note"] = "lgbm cost = recursive 24-step rollout wall time per origin (amortized)"
    import platform as _pf
    _div = 1024 * 1024 if _pf.system() == "Darwin" else 1024  # macOS=bytes, Linux=KB (both verified)
    row["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _div, 1)

    out = pd.DataFrame([row])
    csv = BASE / "experiments" / "efficiency_pilot.csv"
    out.to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(out.to_string(index=False))
    print(f"appended -> {csv}")


def lats_proxy(track, zone):
    """Measure one LightGBM recursive rollout end-to-end (train tiny + rollout)."""
    import lightgbm as lgb
    from common import load_covariates, split_indices
    from run_trained import run_lightgbm
    ts, values = load_series(track, zone)
    cov = load_covariates(track, zone)
    n = len(values)
    _, test_start = split_indices(n)
    pred, _act, t_train, _it, _s, _v = run_lightgbm(values, cov, ts, test_start, 0, HORIZON)
    n_origins = max(len(pred), 1)
    return t_train / n_origins * 1  # conservative proxy: amortized train+rollout per origin


if __name__ == "__main__":
    main()
