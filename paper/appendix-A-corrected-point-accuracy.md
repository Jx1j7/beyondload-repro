# Appendix A. Corrected per-site point accuracy

Generated from the archived 397 deduplicated `step=1` runs by `code/experiments/recalculate_metrics_cpu.py` and `code/experiments/build_point_appendix_cpu.py`. The original `experiments/results.csv` is unchanged.

Scores are corrected MASE (lower is better): archived all-hour MAE for wind/load or observed-power >1% MAE for solar, divided by the site's 24-hour seasonal-naïve MAE in the training period. Each cell is the seed mean; five-seed cells also show the seed standard deviation. A dash means that no `step=1` result is archived for that combination. Seed variation is not a confidence interval over prediction origins or sites.

## Wind (10 zones)

| Site | LightGBM+NWP | TTM | TimesFM | Moirai-2 | PatchTST | TimeMixer | DLinear | Chronos-2 | Chronos-Bolt | iTransformer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.599 ± 0.005 (n=5) | 0.801 | 0.814 | 0.807 | 0.850 ± 0.035 (n=5) | 0.838 ± 0.020 (n=5) | 0.815 ± 0.001 (n=5) | 0.977 | 0.872 | 0.900 ± 0.018 (n=5) |
| 2 | 0.533 ± 0.003 (n=5) | 0.853 | 0.843 | 0.846 | 0.903 ± 0.010 (n=5) | 0.910 ± 0.020 (n=5) | 0.893 ± 0.001 (n=5) | 0.861 | 0.904 | 0.978 ± 0.012 (n=5) |
| 3 † | 0.396 ± 0.005 (n=5) | 0.720 | 0.779 | 0.773 | 0.762 ± 0.007 (n=5) | 0.774 ± 0.019 (n=5) | 0.803 ± 0.000 (n=5) | 0.783 | 0.840 | 0.880 ± 0.028 (n=5) |
| 4 | 0.427 ± 0.009 (n=5) | 0.852 | 0.875 | 0.865 | 0.903 ± 0.018 (n=5) | 0.901 ± 0.017 (n=5) | 0.909 ± 0.001 (n=5) | 0.931 | 0.937 | 0.930 ± 0.016 (n=5) |
| 5 | 0.436 ± 0.006 (n=5) | 0.851 | 0.884 | 0.856 | 0.876 ± 0.012 (n=5) | 0.894 ± 0.011 (n=5) | 0.897 ± 0.000 (n=5) | 0.926 | 0.962 | 0.920 ± 0.023 (n=5) |
| 6 | 0.424 ± 0.002 (n=5) | 0.814 | 0.847 | 0.823 | 0.834 ± 0.005 (n=5) | 0.850 ± 0.019 (n=5) | 0.860 ± 0.001 (n=5) | 0.887 | 0.912 | 0.884 ± 0.006 (n=5) |
| 7 | 0.449 ± 0.004 (n=5) | 0.813 | 0.823 | 0.819 | 0.821 ± 0.009 (n=5) | 0.836 ± 0.012 (n=5) | 0.844 ± 0.000 (n=5) | 0.848 | 0.886 | 0.903 ± 0.018 (n=5) |
| 8 | 0.546 ± 0.003 (n=5) | 0.784 | 0.801 | 0.785 | 0.848 ± 0.031 (n=5) | 0.813 ± 0.005 (n=5) | 0.826 ± 0.000 (n=5) | 0.809 | 0.862 | 0.893 ± 0.006 (n=5) |
| 9 | 0.435 ± 0.002 (n=5) | 0.794 | 0.810 | 0.793 | 0.840 ± 0.033 (n=5) | 0.819 ± 0.014 (n=5) | 0.846 ± 0.000 (n=5) | 0.823 | 0.848 | 0.882 ± 0.020 (n=5) |
| 10 | 0.472 ± 0.005 (n=5) | 0.866 | 0.888 | 0.861 | 0.906 ± 0.006 (n=5) | 0.923 ± 0.009 (n=5) | 0.903 ± 0.000 (n=5) | 0.898 | 0.915 | 0.950 ± 0.005 (n=5) |

## Solar (3 sites)

| Site | LightGBM+NWP | TTM | TimesFM | Moirai-2 | PatchTST | TimeMixer | DLinear | Chronos-2 | Chronos-Bolt | iTransformer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.612 ± 0.007 (n=5) | 0.985 | 1.013 | 0.919 | 1.274 ± 0.123 (n=5) | — | — | 0.896 | 0.914 | 1.040 ± 0.016 (n=5) |
| 2 | 0.602 ± 0.004 (n=5) | 0.940 | 0.992 | 0.859 | 1.128 ± 0.064 (n=5) | 1.020 ± 0.013 (n=5) | 1.003 ± 0.004 (n=5) | 0.835 | 0.846 | 0.957 ± 0.007 (n=5) |
| 3 | 0.601 ± 0.003 (n=5) | 0.934 | 1.002 | 0.883 | 1.154 ± 0.057 (n=5) | 1.179 ± 0.317 (n=5) | 0.977 ± 0.004 (n=5) | 0.854 | 0.872 | 0.967 ± 0.004 (n=5) |

## Load (1 zone)

| Site | LightGBM+NWP | TTM | TimesFM | Moirai-2 | PatchTST | TimeMixer | DLinear | Chronos-2 | Chronos-Bolt | iTransformer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.378 ± 0.003 (n=5) | — | — | — | 0.960 ± 0.036 (n=5) | — | — | 0.947 | 0.756 | 1.420 ± 0.039 (n=5) |

† Wind zone 3 has missing target intervals; the eligible-origin sets differ across model runners and context lengths. Its values are descriptive, not strictly paired model comparisons. The thirteen-site-dataset sensitivity test excluding this zone is reported in Supplementary Section S1 and `experiments/metric_audit/report.md`.

The complete per-run corrected values, site denominator, and site-level mean CSVs remain in `experiments/metric_audit/`. The archive has no per-origin forecasts or errors, so hourly paired confidence intervals and a common-origin zone-3 test cannot be reconstructed without additional inference.
