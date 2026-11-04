"""Audit all-zones main experiment coverage against protocol v1.0.

Checks experiments/results.csv for every expected combination:
  - zero-shot: 5 models x 12 zones (wind 1-10, solar 2-3; solar z1 preexisting) step=1
  - trained:   5 models x 5 seeds x 12 zones, step=1
Prints MISSING combos in a machine-readable list for the rerun loop.
Usage: python audit_allzones.py [--csv results.csv]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
ZONES = [("wind", z) for z in range(1, 11)] + [("solar", z) for z in (2, 3)]
ZS_MODELS = ["chronos", "chronos2", "ttm", "moirai", "timesfm"]
TR_MODELS = ["lightgbm+cov", "dlinear", "patchtst+cov", "itransformer+cov", "timemixer+cov"]
SEEDS = [0, 1, 2, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(BASE / "experiments/results.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    # only count rows actually at step=1 (protocol formal); batched or serial either
    step1 = df[df.note.fillna("").str.contains("step=1")]
    have = set(zip(step1.model, step1.dataset))

    missing = []
    for m in ZS_MODELS:
        for t, z in ZONES:
            if (m, f"gefcom2014_{t}_zone{z}") not in have:
                missing.append(("zs", m, t, z, -1))
    for m in TR_MODELS:
        for t, z in ZONES:
            row = step1[(step1.model == m) & (step1.dataset == f"gefcom2014_{t}_zone{z}")]
            got_seeds = set(row.seed.astype(int))
            for s in SEEDS:
                if s not in got_seeds:
                    missing.append(("tr", m, t, z, s))

    print(f"[audit] step=1 rows: {len(step1)} | expected zs={len(ZS_MODELS)*len(ZONES)}"
          f" tr={len(TR_MODELS)*len(SEEDS)*len(ZONES)}")
    print(f"[audit] MISSING: {len(missing)}")
    for kind, m, t, z, s in missing:
        print(f"MISSING {kind} {m} {t} {z} {s}")
    dup = step1.duplicated(subset=["model", "dataset", "seed"], keep=False)
    print(f"[audit] duplicate (model,dataset,seed) rows: {int(dup.sum())}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
