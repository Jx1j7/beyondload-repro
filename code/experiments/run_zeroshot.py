"""Zero-shot TSFM zoo runner: one model per process invocation (8GB RAM -> process isolation).

Usage: .venv/bin/python code/experiments/run_zeroshot.py --model {chronos|timesfm|moirai|ttm}
                                               [--track solar --zone 1] [--step 24] [--covariates]

Context policy: TSFMs get 512-step context (within every model's native range);
specialized models use the protocol's 168-step lookback. Context length is
recorded in the results note. Point forecasts: median of the predictive
distribution. --covariates is supported for Moirai-2 and Chronos-2;
Chronos-Bolt has no covariate input. The historical Chronos-2 +cov pilot used
an earlier future-row construction and should not be reused as a valid
future-covariate result; see competition_aligned_cpu for the corrected arm.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HORIZON, append_results, collect_targets, load_covariates,
                    load_series, metrics, rolling_origins, seasonal_naive_denom, split_indices)

ZS_CONTEXT = 512  # TSFM context; specialized models use protocol LOOKBACK=168


def origins_of(n_values, test_start, step):
    return list(rolling_origins(n_values - test_start, step))


def forecast_chronos(values, test_start, step, cov=None):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu")
    preds, lats = [], []
    for o in origins_of(len(values), test_start, step):
        g = test_start + o
        ctx = torch.tensor(values[g - ZS_CONTEXT:g], dtype=torch.float32)
        t0 = time.time()
        q = pipe.predict(inputs=ctx, prediction_length=HORIZON)  # [1, Q, H]
        lats.append(time.time() - t0)
        preds.append(q[0].median(dim=0).values.numpy())
    return np.concatenate(preds), float(np.median(lats)), "context=512"


def forecast_timesfm(values, test_start, step, cov=None):
    import timesfm
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu", per_core_batch_size=1, horizon_len=128,
            context_len=ZS_CONTEXT, num_layers=50,
            use_positional_embedding=False),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
    )
    preds, lats = [], []
    for o in origins_of(len(values), test_start, step):
        g = test_start + o
        t0 = time.time()
        pts, _ = model.forecast([values[g - ZS_CONTEXT:g]])
        lats.append(time.time() - t0)
        preds.append(np.asarray(pts[0])[:HORIZON])
    return np.concatenate(preds), float(np.median(lats)), "context=512"


def forecast_moirai(values, test_start, step, cov=None):
    import pandas as pd
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    repo = "Salesforce/moirai-2.0-R-small"
    cov_dim = 0 if cov is None else cov.shape[1]
    module = Moirai2Module.from_pretrained(repo)
    model = Moirai2Forecast(
        module=module, prediction_length=HORIZON, context_length=ZS_CONTEXT,
        target_dim=1, feat_dynamic_real_dim=cov_dim, past_feat_dynamic_real_dim=0)
    predictor = model.create_predictor(batch_size=1)
    preds, lats = [], []
    for o in origins_of(len(values), test_start, step):
        g = test_start + o
        entry = {"start": pd.Period("2024-01-01", freq="h"),
                 "target": values[g - ZS_CONTEXT:g], "item_id": "t"}
        if cov is not None:
            # NWP known ahead: spans context + horizon (past+future dynamic features)
            entry["feat_dynamic_real"] = cov[g - ZS_CONTEXT:g + HORIZON]
        t0 = time.time()
        fcst = next(iter(predictor.predict([entry])))
        lats.append(time.time() - t0)
        preds.append(np.asarray(fcst.median)[:HORIZON])
    return (np.concatenate(preds), float(np.median(lats)),
            f"context=512;repo={repo};cov={cov_dim}")


def forecast_ttm(values, test_start, step, cov=None):
    from tsfm_public import TinyTimeMixerForPrediction
    model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM", revision="main")
    model.eval()
    preds, lats = [], []
    for o in origins_of(len(values), test_start, step):
        g = test_start + o
        ctx = torch.tensor(values[g - ZS_CONTEXT:g], dtype=torch.float32).reshape(1, -1, 1)
        t0 = time.time()
        with torch.no_grad():
            out = model(past_values=ctx).prediction_outputs  # [1, H, 1]
        lats.append(time.time() - t0)
        preds.append(out[0, :HORIZON, 0].numpy())
    return np.concatenate(preds), float(np.median(lats)), "context=512"


def forecast_chronos2(values, test_start, step, cov=None):
    """Chronos-2: use predict_df's future_df for known future covariates."""
    import pandas as pd
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2")
    preds, lats = [], []
    for o in origins_of(len(values), test_start, step):
        g = test_start + o
        idx = pd.date_range("2024-01-01", periods=ZS_CONTEXT + HORIZON, freq="h")
        d = {"item_id": "a", "timestamp": idx[:ZS_CONTEXT],
             "target": values[g - ZS_CONTEXT:g]}
        future_df = None
        if cov is not None:
            block = cov[g - ZS_CONTEXT:g + HORIZON]
            for i in range(cov.shape[1]):
                d[f"cov{i}"] = block[:ZS_CONTEXT, i]
            future_df = pd.DataFrame({"item_id": "a", "timestamp": idx[ZS_CONTEXT:]})
            for i in range(cov.shape[1]):
                future_df[f"cov{i}"] = block[ZS_CONTEXT:, i]
        df = pd.DataFrame(d)
        t0 = time.time()
        out = pipe.predict_df(df, future_df=future_df,
                              prediction_length=HORIZON)
        lats.append(time.time() - t0)
        preds.append(out["predictions"].to_numpy()[:HORIZON])
    tag = "cov" if cov is not None else "nocov"
    return np.concatenate(preds), float(np.median(lats)), f"chronos-2;{tag}"


MODELS = {"chronos": forecast_chronos, "chronos2": forecast_chronos2, "timesfm": forecast_timesfm,
          "moirai": forecast_moirai, "ttm": forecast_ttm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--track", default="solar")
    ap.add_argument("--zone", type=int, default=1)
    ap.add_argument("--step", type=int, default=HORIZON)
    ap.add_argument("--covariates", action="store_true")
    args = ap.parse_args()
    if args.covariates and args.model not in ("moirai", "chronos2"):
        ap.error("--covariates only implemented for moirai and chronos2")

    ts, values = load_series(args.track, args.zone)
    cov = load_covariates(args.track, args.zone).to_numpy() if args.covariates else None
    n = len(values)
    _, test_start = split_indices(n)
    denom = seasonal_naive_denom(values, test_start, args.step)

    t0 = time.time()
    pred, lat, note = MODELS[args.model](values, test_start, args.step, cov)
    # filter out origins whose 24-step target contains NaN (protocol >3h-drop rule);
    # everything downstream stays 2-D: (n_origins, HORIZON)
    origins = list(rolling_origins(n - test_start, args.step))
    pred = np.asarray(pred).reshape(len(origins), HORIZON)
    finite = np.array([not np.isnan(values[test_start + o:test_start + o + HORIZON]).any()
                       for o in origins])
    pred = pred[finite]
    act = np.stack([values[test_start + o:test_start + o + HORIZON]
                    for o, k in zip(origins, finite) if k])
    assert pred.shape == act.shape, f"pred {pred.shape} vs targets {act.shape}"
    skipped = int((~finite).sum())
    m = metrics(pred, act, denom)
    cov_tag = "+cov" if args.covariates else ""
    append_results([{**m, "model": args.model + cov_tag,
                     "dataset": f"gefcom2014_{args.track}_zone{args.zone}",
                     "horizon": HORIZON, "seed": -1, "time_s": round(lat, 4),
                     "note": f"{note};step={args.step};skipped={skipped}",
                     "wall_s": round(time.time() - t0, 1), "date": "2026-10-29"}])
    gc.collect()


if __name__ == "__main__":
    main()
