# Appendix B. Competition-schedule-aligned paired point accuracy

The daily paired-loss source is `experiments/competition_aligned_cpu/paired_origin_losses.csv` (SHA-256 `1694ff3474d7eca94c8d81d37eaaa1c649c2cf919caf03e41b7887bbdc9272d6`). Its `count` column records scored target hours per origin, required to reconstruct solar site scores from the origin-level MASE values. Exploratory paired intervals and their limitations are documented in `paired_site_intervals_method.md` in the same directory.

This post-hoc CPU analysis uses the same eligible daily 24-hour forecast windows for both models. Source weather records run from 01:00 to next-day 00:00; the midnight issue time is inferred from the GEFCom2014 publication and is not an explicit file field. A window is retained only when its preceding 512-hour raw target context, next 24 raw target hours, and weather fields are complete. The old hourly-origin scores in Appendix A are a different protocol.

LightGBM is site-trained with five seeds; scores below average seed scores. Chronos-2 is zero-shot at pinned local revision `29ec3766d36d6f73f0696f85560a422f50e8498c`. Both models receive future-known calendar in both arms. The NWP arms both receive the next 24 hours' weather fields; Chronos-2 additionally sees their past trajectory through its native interface. Wind is scored over all hours, solar over observed power >1% hours; both use a training-period 24-hour seasonal-naive denominator.

| Site | Days | LightGBM calendar | Chronos-2 calendar | LightGBM + NWP | Chronos-2 + NWP | LightGBM lower by |
|---|---:|---:|---:|---:|---:|---:|
| gefcom2014_solar_zone1 | 157 | 0.804 | 0.749 | 0.579 | 0.687 | 15.7% |
| gefcom2014_solar_zone2 | 157 | 0.817 | 0.695 | 0.575 | 0.634 | 9.3% |
| gefcom2014_solar_zone3 | 157 | 0.772 | 0.716 | 0.565 | 0.626 | 9.8% |
| gefcom2014_wind_zone1 | 52 | 0.805 | 0.825 | 0.610 | 0.704 | 13.3% |
| gefcom2014_wind_zone10 | 138 | 0.870 | 0.836 | 0.474 | 0.682 | 30.5% |
| gefcom2014_wind_zone2 | 86 | 0.864 | 0.824 | 0.565 | 0.672 | 15.9% |
| gefcom2014_wind_zone3 | 60 | 0.676 | 0.690 | 0.393 | 0.575 | 31.7% |
| gefcom2014_wind_zone4 | 138 | 0.837 | 0.833 | 0.425 | 0.598 | 28.9% |
| gefcom2014_wind_zone5 | 138 | 0.862 | 0.858 | 0.440 | 0.669 | 34.3% |
| gefcom2014_wind_zone6 | 138 | 0.825 | 0.812 | 0.430 | 0.637 | 32.5% |
| gefcom2014_wind_zone7 | 138 | 0.829 | 0.831 | 0.450 | 0.642 | 30.0% |
| gefcom2014_wind_zone8 | 138 | 0.822 | 0.821 | 0.550 | 0.669 | 17.8% |
| gefcom2014_wind_zone9 | 138 | 0.782 | 0.795 | 0.437 | 0.628 | 30.5% |

The percentage in the last column is relative to that site's Chronos-2 + NWP score. It is a descriptive effect, not a confidence bound. Wind zones share one source, and only three solar sites are available. Point forecasts, origin-level paired losses, exclusions, input hashes and execution manifests are in `experiments/competition_aligned_cpu/`.
