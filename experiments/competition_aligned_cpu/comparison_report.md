# Competition-aligned, information-aware CPU comparison

This new comparison scores exactly the same daily 01:00–next-day 00:00 target windows for each model, after excluding any window whose raw 512-hour target context or raw 24-hour target has missing values. Weather is assigned to the inferred midnight issue block from the original competition schedule. The source files do not contain an independently recorded issue-time field.

Both models use target history and known calendar. The weather arms add the same target-hour NWP; Chronos-2 receives the past weather sequence through its native covariate interface, while LightGBM uses target-hour weather features. Thus future-information access and scored origins are matched, but context length, feature representation, and training regime differ. Chronos-2 was loaded at a pinned local revision. LightGBM has five site-trained seeds.

Scores use the same training-period 24-hour seasonal-naive denominator as the metric audit; wind uses all target hours and solar only observed power >1% hours. These results are separate from original `results.csv` and its hourly-origin Table 3.

- Paired eligible origins: 1635 across 13 renewable sites.
- Weather-arm site wins: {'LightGBM': 13}.
- Sensitivity excluding sparse-context wind zones 1–3: on wind zones 4–10, LightGBM + NWP mean 0.458 versus Chronos-2 + NWP 0.647; LightGBM lower at all seven sites.
- Origin-loss CSV SHA-256: `7b3f0f34278512c343df2fb8125d98273ef98b70574ee4ca8d8b4e4fa7240f45`.

| Track | Model | Calendar only | Calendar + NWP | Sites |
|---|---|---:|---:|---:|
| wind | LightGBM | 0.817 | 0.477 | 10 |
| wind | Chronos-2 | 0.813 | 0.648 | 10 |
| solar | LightGBM | 0.798 | 0.573 | 3 |
| solar | Chronos-2 | 0.720 | 0.649 | 3 |

## Per-site corrected MASE

| Site | Origins | LightGBM calendar | Chronos-2 calendar | LightGBM + NWP | Chronos-2 + NWP |
|---|---:|---:|---:|---:|---:|
| gefcom2014_solar_zone1 | 157 | 0.804 | 0.749 | 0.579 | 0.687 |
| gefcom2014_solar_zone2 | 157 | 0.817 | 0.695 | 0.575 | 0.634 |
| gefcom2014_solar_zone3 | 157 | 0.772 | 0.716 | 0.565 | 0.626 |
| gefcom2014_wind_zone1 | 52 | 0.805 | 0.825 | 0.610 | 0.704 |
| gefcom2014_wind_zone10 | 138 | 0.870 | 0.836 | 0.474 | 0.682 |
| gefcom2014_wind_zone2 | 86 | 0.864 | 0.824 | 0.565 | 0.672 |
| gefcom2014_wind_zone3 | 60 | 0.676 | 0.690 | 0.393 | 0.575 |
| gefcom2014_wind_zone4 | 138 | 0.837 | 0.833 | 0.425 | 0.598 |
| gefcom2014_wind_zone5 | 138 | 0.862 | 0.858 | 0.440 | 0.669 |
| gefcom2014_wind_zone6 | 138 | 0.825 | 0.812 | 0.430 | 0.637 |
| gefcom2014_wind_zone7 | 138 | 0.829 | 0.831 | 0.450 | 0.642 |
| gefcom2014_wind_zone8 | 138 | 0.822 | 0.821 | 0.550 | 0.669 |
| gefcom2014_wind_zone9 | 138 | 0.782 | 0.795 | 0.437 | 0.628 |

No population-level significance is claimed: wind sites share one competition source, solar has three sites, and adjacent daily losses may be serially correlated. Per-origin paired losses are saved for further analysis without inventing hourly uncertainty from aggregate errors.
