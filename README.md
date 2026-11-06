# Beyond load: reproducibility package

This repository preserves the code and numerical artifacts for a **post-hoc, competition-schedule-aligned** comparison of LightGBM and Chronos-2 on the wind and solar tracks of GEFCom2014. The manuscript's earlier hourly multi-model sweep is retained as retrospective context. No claim of independently timestamped operational NWP, population-level significance, or model-architecture causality follows from these files.

## What is included

- `code/experiments/`: experiment and audit source, including the four-arm daily comparison.
- `experiments/competition_aligned_cpu/`: eligible daily origins, 156 point-forecast CSVs, site/origin scores, per-origin scored-hour counts, exploratory within-site paired intervals, horizon profiles, common-date sensitivity, capacity-normalized error summaries, and SHA-256 run manifests.
- `experiments/results.csv` and `experiments/metric_audit/`: original aggregate archive and the separately corrected metric audit. The archived file is unchanged (SHA-256 `de787fa9079cab02598904398344166142d832ac734ec918e0435999c275dea8`).
- `paper/appendix-*.md`: per-site appendices from the archived analyses.
- `paper/submission/`: the current supplementary analyses and Figures 1–2 / Figures S1–S2 in PNG, PDF, EPS, and SVG. The submitted manuscript itself is not included.
- `code/experiments/make_submission_figures_cpu.py`: regenerates the current figures from the archived CSVs without model inference.
- `docs/05-RQ与实验协议.md`: original protocol plus explicitly dated post-hoc amendment log. Its instructions are historical research records, not repository commands.

The original public competition data, processed data, model checkpoints, virtual environment, execution logs, and the unpublished manuscript are **not** redistributed. The local Hugging Face cache path in the Chronos-2 manifest is redacted for release; the model revision and all prediction-file hashes are preserved.

## Inspect the archived results without running models

Use Python 3.11 or newer:

```bash
python scripts/verify_release.py
```

The expected result is 13 site-series, 1,635 shared daily forecast origins, 156 prediction files, and 39,240 shared observed target values. `experiments/competition_aligned_cpu/comparison_report.md` summarizes the corrected MASE results; `paper/appendix-B-competition-aligned.md` gives per-site scores. `paired_origin_losses.csv` includes `count`, the number of scored target hours per origin; weight solar origin MASE by this count to reconstruct site scores. `paired_site_intervals_method.md` records the exploratory block-bootstrap procedure and its limitations.

The current Figures 1–2 and supplementary Figures S1–S2 can be regenerated after installing `requirements-core.txt` with:

```bash
.venv/bin/python code/experiments/make_submission_figures_cpu.py
```

The figures read `paired_per_site.csv`, `horizon_summary.csv`, `corrected_per_site.csv`, and `per_site_comparison.csv`. Generate the post-hoc horizon and sensitivity tables first with `code/experiments/derive_submission_sensitivities_cpu.py`; this reads the archived point forecasts and does not run either model. Figure S1 is descriptive; no significance bar is plotted. Figure S2 shows unconnected points because its two arms need not have identical eligible origins. Earlier figure-generating scripts remain as historical research code, but their superseded outputs are omitted from this release.

## Reproduce the CPU comparison from source data

1. Obtain the original GEFCom2014 supplementary archive from [Hong et al. (2016), International Journal of Forecasting](https://www.sciencedirect.com/science/article/pii/S0169207016000133). Save it as `data/raw/1-s2.0-S0169207016000133-mmc1.zip`. The expected SHA-256 is `d68d957270edd93b26a37d0f9b5e901f942abdf34c75eacbe14e417beb16e154`. The original [competition paper](https://robjhyndman.com/papers/gefcom2014.pdf) describes the daily midnight NWP schedule.
2. Create an environment. `requirements-core.txt` contains the versions needed for the new daily comparison; `requirements.lock.txt` is the broader original environment snapshot. On macOS, LightGBM may need Homebrew `libomp`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python scripts/prepare_gefcom2014.py
.venv/bin/python code/data_scripts/assemble_gefcom2014.py
.venv/bin/python code/data_scripts/clean_gefcom2014.py
```

3. Download the pinned Chronos-2 checkpoint into the Hugging Face cache. The run script intentionally uses `local_files_only=True` to prevent a moving model revision during inference.

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('amazon/chronos-2', revision='29ec3766d36d6f73f0696f85560a422f50e8498c')"
```

4. Run the analysis in this order. All inference runs on CPU; the archived Chronos-2 run took about 12 minutes and LightGBM about 5 minutes on the recorded macOS host. Run times vary by machine. These scripts write into the archived output directory; use a fresh checkout if you want to retain the bundled point forecasts untouched.

```bash
.venv/bin/python code/experiments/audit_competition_issue_schedule_cpu.py
.venv/bin/python code/experiments/run_competition_lightgbm_cpu.py
.venv/bin/python code/experiments/run_competition_chronos2_cpu.py
.venv/bin/python code/experiments/analyze_competition_aligned_cpu.py
.venv/bin/python code/experiments/make_competition_figure_cpu.py
.venv/bin/python code/experiments/build_competition_appendix_cpu.py
```

The new scripts write outside the original `experiments/results.csv`. For the archived hourly results, `code/experiments/recalculate_metrics_cpu.py` derives corrected scaled scores from the retained aggregate MAE and freshly prepared target series. It does **not** re-run or validate historical model checkpoints. Exact reconstruction of the old multi-model sweep is limited because resolved checkpoint revisions and reliable first-run timestamps were not recorded per result; the new four-arm experiment has its own pinned revision and point forecasts.

## Protocol and limits

The daily origin starts at 01:00 and covers the next 24 target hours. GEFCom2014's paper says NWP was issued at midnight, and the source weather records appear in matching daily 01:00–next-day 00:00 blocks. The files do not contain explicit per-record issue times or forecast vintages. Eligible test windows require the preceding 512 hours and following 24 hours of **raw** target values, plus complete weather fields, so target interpolation cannot affect their context or scoring window.

Both models receive target history and known future calendar. The NWP arms both receive the next 24 hours' weather fields. Chronos-2 additionally receives past weather through `future_df`'s matching past covariates; LightGBM uses target-hour weather features. Consequently, the Chronos-2 weather-minus-calendar improvement cannot isolate the value of future NWP alone. Context length, model training, and feature representation are different. Wind zones come from one competition source, and there are only three solar sites. Results are descriptive for these sites. The source-file time zone and local clock alignment are not independently verified.

The old `chronos2+cov;step=24` rows in `experiments/results.csv` are **not** future-NWP evidence: the old API call did not pass `future_df`. The daily comparison uses the correct API and pins checkpoint revision `29ec3766d36d6f73f0696f85560a422f50e8498c`.

## Third-party software and data

The earlier neural-model runners depend on [Time-Series-Library](https://github.com/thuml/Time-Series-Library) at commit `4e938a1767106324dd753b2a44832bf870a0252e`. Its code is not vendored here; install it separately under `code/tslib` to execute those historical runners. Chronos-2 weights are downloaded separately from the model publisher. Original GEFCom data should be obtained from its publisher. This repository does not grant a license to third-party data or model weights.

The original code and documentation in this repository are released under the [MIT License](LICENSE). This license does not apply to the original GEFCom2014 data, pretrained model weights, or third-party software, which retain their own terms. Archived prediction files and derived numerical outputs are supplied for verification; users should cite the paper and original data source when reusing them.
