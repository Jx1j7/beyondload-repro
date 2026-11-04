# Chronos-2 competition-aligned CPU manifest

- Started UTC: 2026-11-04T07:30:15.211707+00:00
- Finished UTC: 2026-11-04T07:42:23.072662+00:00
- Model: amazon/chronos-2 at pinned revision `29ec3766d36d6f73f0696f85560a422f50e8498c`
- Local snapshot: local Hugging Face cache path redacted for release
- Python: 3.11.15; Torch: 2.4.1
- Host: macOS-15.6-arm64-arm-64bit; Torch CPU threads: 4
- Source script SHA-256: `19f77e33d17c560247d50164f807340a23b4b762bbda941062e80d13f7941ae6`
- Candidate origins SHA-256: `c187ca1cde7d8876f9255f28e36f4b99fb3962ffc62e89721122751c698f420e`
- Both arms: 512-hour target history and future-known calendar; weather arm additionally receives past and next-24-hour NWP through the Chronos-2 `future_df` API.
- Daily issue time is inferred from competition schedule, not a source-file issue-time column. Original `results.csv` untouched.

| Dataset | Arm | Origins | CPU seconds | File | SHA-256 |
|---|---|---:|---:|---|---|
| gefcom2014_solar_zone1 | calendar | 157 | 14.39 | `gefcom2014_solar_zone1_calendar_points.csv` | `6b4e30ddc5032f73dc52c49be8e96c997f55d63deaf0fa028fb57d34a4b2789a` |
| gefcom2014_solar_zone1 | weather | 157 | 44.54 | `gefcom2014_solar_zone1_weather_points.csv` | `4bfceaa50f2768368b52eb544a448da1a1c939cc7ad7d052f05f8ad90db08b2d` |
| gefcom2014_solar_zone2 | calendar | 157 | 14.96 | `gefcom2014_solar_zone2_calendar_points.csv` | `ac2b6122d7a6654e60843543fe4d959516f74c18db6eb48dbcb87f5819f76e36` |
| gefcom2014_solar_zone2 | weather | 157 | 60.99 | `gefcom2014_solar_zone2_weather_points.csv` | `8e4d4a3237415deb9ef213701b6767a5766fb2a81a0b49a1f78e709e14351391` |
| gefcom2014_solar_zone3 | calendar | 157 | 19.98 | `gefcom2014_solar_zone3_calendar_points.csv` | `5f104a69b71f803515594291cf4c4fbcd30333e8f89b9dc124c7ccf2d2a07154` |
| gefcom2014_solar_zone3 | weather | 157 | 65.21 | `gefcom2014_solar_zone3_weather_points.csv` | `76a836e3e0b8c91b3a1da53c1d71188d74a0ba6a3d3e917f5eb84be8290c412e` |
| gefcom2014_wind_zone1 | calendar | 52 | 6.72 | `gefcom2014_wind_zone1_calendar_points.csv` | `81bfc00c28ad6b4571947f85e2a622f27a07f3895bb7e2ffab85ad2f17b98236` |
| gefcom2014_wind_zone1 | weather | 52 | 11.64 | `gefcom2014_wind_zone1_weather_points.csv` | `6388f54a5567e437067897786b20bc507860da6d3b4430e4b25647a5344155ec` |
| gefcom2014_wind_zone10 | calendar | 138 | 17.03 | `gefcom2014_wind_zone10_calendar_points.csv` | `62806bbce1fb102b4f3694f4af8b336e649111a083508e61a161514808b2ac4d` |
| gefcom2014_wind_zone10 | weather | 138 | 31.27 | `gefcom2014_wind_zone10_weather_points.csv` | `e9f21523935c4a542fff5ff8b24e93d34ea61b41d8dad10113e6dad2cf537dc8` |
| gefcom2014_wind_zone2 | calendar | 86 | 11.82 | `gefcom2014_wind_zone2_calendar_points.csv` | `022b0e996ee0fdf5ed22042c43d9569e6a1e811ecb3359edfde7ff94104ff802` |
| gefcom2014_wind_zone2 | weather | 86 | 20.17 | `gefcom2014_wind_zone2_weather_points.csv` | `c1c6e92f6d0414e33c9520db4acf81d1abb9e5db0dd7dbec63b839754ef7317a` |
| gefcom2014_wind_zone3 | calendar | 60 | 8.67 | `gefcom2014_wind_zone3_calendar_points.csv` | `9fcc8d9877340083a587de1b71fc122cb05dbb1a49cfbc42d0ea645821a26551` |
| gefcom2014_wind_zone3 | weather | 60 | 20.86 | `gefcom2014_wind_zone3_weather_points.csv` | `500bd27a75a26cb1fa903f3a58f565705088120c73a424e1b70a54f0da358839` |
| gefcom2014_wind_zone4 | calendar | 138 | 24.67 | `gefcom2014_wind_zone4_calendar_points.csv` | `c20c49d50ee924c5b13df459194dae17a613cc47fbc0233628f68e6aadb3ad9e` |
| gefcom2014_wind_zone4 | weather | 138 | 37.54 | `gefcom2014_wind_zone4_weather_points.csv` | `a471581a827aa28b041bc17645c4ceb317c5218a3374a424da3e955c97b3c14f` |
| gefcom2014_wind_zone5 | calendar | 138 | 20.41 | `gefcom2014_wind_zone5_calendar_points.csv` | `a3ddcc065328abdffc6569cc962b910d04c6187b5e7379267d63981c048dd107` |
| gefcom2014_wind_zone5 | weather | 138 | 32.89 | `gefcom2014_wind_zone5_weather_points.csv` | `11e63400cd0e549fb94b1d4a39a0e13878bcfc8049d4646bbe379a79fe8c7e9e` |
| gefcom2014_wind_zone6 | calendar | 138 | 19.65 | `gefcom2014_wind_zone6_calendar_points.csv` | `f1b871c9946c87c13fb0b84779203a5e1135fe07c8611bbb83d5b545beeb453d` |
| gefcom2014_wind_zone6 | weather | 138 | 44.69 | `gefcom2014_wind_zone6_weather_points.csv` | `9f7bd5a5ca9b89edf2456b6d0b52ad5d5c2f93aadfa67ca474fc5350fd9551c3` |
| gefcom2014_wind_zone7 | calendar | 138 | 30.93 | `gefcom2014_wind_zone7_calendar_points.csv` | `54b87e0654e549610dea215eaa8dc3a518780a3590b31e3ef710a55043c0f285` |
| gefcom2014_wind_zone7 | weather | 138 | 56.55 | `gefcom2014_wind_zone7_weather_points.csv` | `c349de1c96464f36567db6fb8e82e31307356a399987a9926ba51f40fce964aa` |
| gefcom2014_wind_zone8 | calendar | 138 | 28.85 | `gefcom2014_wind_zone8_calendar_points.csv` | `f7c79d9d2035e9baec79cdb10a3ae7682b481fd14d781e69bf6125196b7055cc` |
| gefcom2014_wind_zone8 | weather | 138 | 33.62 | `gefcom2014_wind_zone8_weather_points.csv` | `311b56b565264fade86ebd371d5765fac65e30083e7b9a52469fc55693f45ab2` |
| gefcom2014_wind_zone9 | calendar | 138 | 16.93 | `gefcom2014_wind_zone9_calendar_points.csv` | `0ca970ca7dc00f51c091a4fabf86c103214b9161f18ec4deb83a4a46e58afd2f` |
| gefcom2014_wind_zone9 | weather | 138 | 29.89 | `gefcom2014_wind_zone9_weather_points.csv` | `96997e1d29319e16da7566e118199f38da5404634945b62be1c26760523605a7` |
