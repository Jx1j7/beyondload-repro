# GEFCom competition-aligned origin audit

The original competition paper specifies daily midnight NWP issuance. The local wind/solar weather records run from 01:00 through next-day 00:00 in contiguous 24-hour blocks (731 days per wind site and 821 days per solar site), so the issue time below is inferred as midnight before the 01:00 valid hour. The files contain no explicit issue-time or forecast-vintage field; this is a competition-schedule alignment, not independently timestamped operational NWP.

For each candidate, the 24-hour target and preceding 512-hour raw target context must be complete. This removes windows touched by the cleaning script's bidirectional interpolation and gives all compared models identical test origins. Weather fields must be finite over context and horizon.

- Candidate daily origins: 1861.
- Eligible common origins: 1635.
- Candidate-origins CSV SHA-256: `c187ca1cde7d8876f9255f28e36f4b99fb3962ffc62e89721122751c698f420e`.

| Track | Site | Candidates | Eligible | Raw context complete | Raw target complete |
|---|---|---:|---:|---:|---:|
| solar | gefcom2014_solar_zone1 | 157 | 157 | 157 | 157 |
| solar | gefcom2014_solar_zone2 | 157 | 157 | 157 | 157 |
| solar | gefcom2014_solar_zone3 | 157 | 157 | 157 | 157 |
| wind | gefcom2014_wind_zone1 | 139 | 52 | 56 | 133 |
| wind | gefcom2014_wind_zone10 | 139 | 138 | 139 | 138 |
| wind | gefcom2014_wind_zone2 | 139 | 86 | 89 | 133 |
| wind | gefcom2014_wind_zone3 | 139 | 60 | 62 | 125 |
| wind | gefcom2014_wind_zone4 | 139 | 138 | 139 | 138 |
| wind | gefcom2014_wind_zone5 | 139 | 138 | 139 | 138 |
| wind | gefcom2014_wind_zone6 | 139 | 138 | 139 | 138 |
| wind | gefcom2014_wind_zone7 | 139 | 138 | 139 | 138 |
| wind | gefcom2014_wind_zone8 | 139 | 138 | 139 | 138 |
| wind | gefcom2014_wind_zone9 | 139 | 138 | 139 | 138 |

Reference: Hong et al., *International Journal of Forecasting* 32 (2016), §§6–7: https://robjhyndman.com/papers/gefcom2014.pdf
