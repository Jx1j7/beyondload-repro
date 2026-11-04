"""Batched zero-shot runner for the all-zones main experiment (protocol v1.0).

Per-origin loops (run_zeroshot.py) spend ~22 h across 12 zones x 5 TSFMs at
step=1; batching origins into single predict calls cuts Chronos ~10x. This
runner batches where the library supports it and MUST reproduce per-origin
outputs (verified by --check_serial against N origins before trusting it).

Usage:
  .venv/bin/python code/experiments/run_zeroshot_batched.py --model chronos \
      [--track wind --zone 3] [--step 1] [--batch 64] [--check_serial 8]
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

ZS_CONTEXT = 512


def batched_chronos(values, origins_g, batch):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu")
    ctxs = [torch.tensor(values[g - ZS_CONTEXT:g], dtype=torch.float32) for g in origins_g]
    preds, t0, n_calls = [], time.time(), 0
    for i in range(0, len(ctxs), batch):
        chunk = torch.stack(ctxs[i:i + batch])  # (b, ctx) — one call, many series
        out = pipe.predict(inputs=chunk, prediction_length=HORIZON)  # (b, Q, H)
        preds.append(out.median(dim=1).values.numpy())
        n_calls += 1
    lat = (time.time() - t0) / len(ctxs)
    return np.concatenate(preds), lat, n_calls


def batched_ttm(values, origins_g, batch):
    from tsfm_public import TinyTimeMixerForPrediction
    model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM", revision="main")
    model.eval()
    ctxs = [values[g - 512:g] for g in origins_g]
    preds, t0, n_calls = [], time.time(), 0
    with torch.no_grad():
        for i in range(0, len(ctxs), batch):
            xb = torch.tensor(np.stack(ctxs[i:i + batch]), dtype=torch.float32).unsqueeze(-1)
            out = model(past_values=xb).prediction_outputs  # (b, H, 1)
            preds.append(out[:, :HORIZON, 0].numpy())
            n_calls += 1
    lat = (time.time() - t0) / len(ctxs)
    return np.concatenate(preds), lat, n_calls


def batched_moirai(values, origins_g, batch):
    import pandas as pd
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    model = Moirai2Forecast(module=module, prediction_length=HORIZON, context_length=ZS_CONTEXT,
                            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    predictor = model.create_predictor(batch_size=batch)
    entries = [{"start": pd.Period("2024-01-01", freq="h"),
                "target": values[g - ZS_CONTEXT:g], "item_id": f"o{i}"}
               for i, g in enumerate(origins_g)]
    preds, t0, n_calls = [], time.time(), 0
    for i in range(0, len(entries), batch):
        fcsts = list(predictor.predict(entries[i:i + batch]))
        preds.extend(np.asarray(f.median) for f in fcsts)
        n_calls += 1
    lat = (time.time() - t0) / len(entries)
    return np.stack(preds), lat, n_calls


def batched_timesfm(values, origins_g, batch):
    import timesfm
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend="cpu", per_core_batch_size=batch,
                                       horizon_len=128, context_len=ZS_CONTEXT,
                                       num_layers=50, use_positional_embedding=False),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
    ctxs = [values[g - ZS_CONTEXT:g] for g in origins_g]
    preds, t0, n_calls = [], time.time(), 0
    for i in range(0, len(ctxs), batch):
        pts, _ = model.forecast(ctxs[i:i + batch])
        preds.extend(np.asarray(p)[:HORIZON] for p in pts)
        n_calls += 1
    lat = (time.time() - t0) / len(ctxs)
    return np.stack(preds), lat, n_calls


def batched_chronos2(values, origins_g, batch):
    import pandas as pd
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2")
    idx = pd.date_range("2024-01-01", periods=ZS_CONTEXT, freq="h")
    preds, t0, n_calls = [], time.time(), 0
    for i in range(0, len(origins_g), batch):
        frames = [pd.DataFrame({"item_id": f"o{j}",
                                "timestamp": idx,
                                "target": values[g - ZS_CONTEXT:g]})
                  for j, g in enumerate(origins_g[i:i + batch], start=i)]
        out = pipe.predict_df(pd.concat(frames), prediction_length=HORIZON)
        # predict_df returns rows in (item_id-major, time) order per item block
        for j in range(len(frames)):
            blk = out[out.item_id == f"o{i + j}"]
            preds.append(blk["predictions"].to_numpy()[:HORIZON])
        n_calls += 1
    lat = (time.time() - t0) / len(origins_g)
    return np.stack(preds), lat, n_calls


MODELS = {"chronos": batched_chronos, "chronos2": batched_chronos2, "ttm": batched_ttm,
          "moirai": batched_moirai, "timesfm": batched_timesfm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--track", default="solar")
    ap.add_argument("--zone", type=int, default=1)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--check_serial", type=int, default=0,
                    help="verify batched == per-origin on first N origins before running")
    args = ap.parse_args()

    ts, values = load_series(args.track, args.zone)
    n = len(values)
    _, test_start = split_indices(n)
    origins = [test_start + o for o in rolling_origins(n - test_start, args.step)
               if not np.isnan(values[test_start + o:test_start + o + HORIZON]).any()
               and not np.isnan(values[test_start + o - 512:test_start + o]).any()]  # NaN context poisons some TSFMs (TTM)

    if args.check_serial:
        sub = origins[:args.check_serial]
        batched, _, _ = MODELS[args.model](values, sub, args.batch)
        serial = []
        for g in sub:
            b1, _, _ = MODELS[args.model](values, [g], 1)
            serial.append(b1[0])
        serial = np.stack(serial)
        max_diff = float(np.abs(batched - serial).max())
        print(f"[check] batch-vs-serial max|diff| = {max_diff:.2e} over {len(sub)} origins")
        assert max_diff < 1e-4, "batched output differs from serial — do not trust batching"

    denom = seasonal_naive_denom(values, test_start, args.step)
    pred, lat, n_calls = MODELS[args.model](values, origins, args.batch)
    act = np.stack([values[g:g + HORIZON] for g in origins])
    assert pred.shape == act.shape
    m = metrics(pred, act, denom)
    append_results([{**m, "model": args.model,
                     "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
                     "horizon": HORIZON, "seed": -1, "time_s": round(lat, 4),
                     "note": f"batched-b{args.batch};ncalls={n_calls};step={args.step}",
                     "wall_s": 0, "date": "2026-11-03"}])


if __name__ == "__main__":
    main()
