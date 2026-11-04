# Post-hoc CPU comparison: LightGBM with and without target-time covariates

This is a new exploratory arm, separate from the original `results.csv` and the five-model main rank test. The 70 no-NWP rows cover 14 site-datasets × 5 seeds at hourly origins. `run_lightgbm_nocov_cpu.py` reuses the original LightGBM training features and hyperparameters, with vectorized recursive prediction. On wind zone 1 and solar zone 1 at step=24, batched and original serial predictions agreed exactly (maximum absolute difference 0).

The no-NWP arm still uses known future calendar features. It is a closer information comparison with target-history TSFMs, but feature sets and training methods are not identical. The with-covariate renewable arm uses retrospective target-time NWP whose issue time was not retained; its gain is not an estimate of deployable day-ahead skill.

| Track | LightGBM no NWP | LightGBM with covariates | Fixed zero-shot comparator |
|---|---:|---:|---:|
| wind | 0.818 | 0.472 | TTM 0.815 |
| solar | 0.937 | 0.605 | Chronos-2 0.862 |
| load | 0.791 | 0.378 | Chronos-Bolt 0.756 |

Within LightGBM, adding target-time weather covariates reduces the corrected cross-site mean MASE by 42.4% on wind and 35.4% on solar. Without NWP, LightGBM and TTM are approximately tied on the wind cross-site mean; Chronos-2 is lower than LightGBM on each of the three solar sites. The load comparison uses temperature covariates rather than renewable NWP.

| Site | No NWP | With covariates | Reduction (%) |
|---|---:|---:|---:|
| gefcom2014_load_zone1 | 0.791 | 0.378 | 52.2 |
| gefcom2014_solar_zone1 | 0.974 | 0.612 | 37.2 |
| gefcom2014_solar_zone2 | 0.941 | 0.602 | 36.0 |
| gefcom2014_solar_zone3 | 0.897 | 0.601 | 33.0 |
| gefcom2014_wind_zone1 | 0.775 | 0.599 | 22.7 |
| gefcom2014_wind_zone10 | 0.870 | 0.472 | 45.7 |
| gefcom2014_wind_zone2 | 0.855 | 0.533 | 37.6 |
| gefcom2014_wind_zone3 | 0.759 | 0.396 | 47.9 |
| gefcom2014_wind_zone4 | 0.863 | 0.427 | 50.5 |
| gefcom2014_wind_zone5 | 0.859 | 0.436 | 49.3 |
| gefcom2014_wind_zone6 | 0.819 | 0.424 | 48.3 |
| gefcom2014_wind_zone7 | 0.807 | 0.449 | 44.4 |
| gefcom2014_wind_zone8 | 0.796 | 0.546 | 31.5 |
| gefcom2014_wind_zone9 | 0.781 | 0.435 | 44.2 |

Wind zone 3 retains the unequal eligible-origin caveat for comparisons across model families. The two LightGBM arms use the same target-window filter in this zone. Results are descriptive; no per-origin paired uncertainty or operational NWP availability test is claimed.
