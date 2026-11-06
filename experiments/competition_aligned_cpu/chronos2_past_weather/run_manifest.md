# Chronos-2 past-weather-only CPU arm

- Started UTC: 2026-11-06T12:09:07.357642+00:00
- Finished UTC: 2026-11-06T12:14:35.496807+00:00
- Model revision: `29ec3766d36d6f73f0696f85560a422f50e8498c`
- Python: 3.11.15; Torch: 2.4.1
- Host: macOS-15.6-arm64-arm-64bit; Torch CPU threads: 4
- Source SHA-256: `656603b20929daa884e3c368bb230db45cb66ade10c2a2e362a41df162a3381c`
- Candidate origins SHA-256: `c187ca1cde7d8876f9255f28e36f4b99fb3962ffc62e89721122751c698f420e`
- Future-known calendar remains in `future_df`. Past NWP fields are only in past input; no future NWP fields are passed to `future_df`.
- Forecasts are separately generated from the pinned checkpoint; no original four-arm prediction files are overwritten.

| Dataset | Origins | CPU seconds | File | SHA-256 |
|---|---:|---:|---|---|
| gefcom2014_solar_zone1 | 157 | 42.63 | `gefcom2014_solar_zone1_past_weather_points.csv` | `4758a5e247c001e674cc51836d73befbe420f65282c95fd761803fd127aa0fe8` |
| gefcom2014_solar_zone2 | 157 | 42.04 | `gefcom2014_solar_zone2_past_weather_points.csv` | `7d155cf823f8b01971aceb7426e0cf6624e2dd98e775e517c6e12c3d870e8076` |
| gefcom2014_solar_zone3 | 157 | 44.21 | `gefcom2014_solar_zone3_past_weather_points.csv` | `102e24fddf51cb1b36565c66848b1d92ecfe906fdd8565e3e310b99243e031cf` |
| gefcom2014_wind_zone1 | 52 | 8.32 | `gefcom2014_wind_zone1_past_weather_points.csv` | `7d7bea48387c47b2b19778efa3adb55e2372dea71959b602de970178a1ac0f8f` |
| gefcom2014_wind_zone10 | 138 | 22.96 | `gefcom2014_wind_zone10_past_weather_points.csv` | `5e636fa78b9eda7032849877d9fbaea1df20021be1b5dcd2048ca3b679d3561a` |
| gefcom2014_wind_zone2 | 86 | 14.3 | `gefcom2014_wind_zone2_past_weather_points.csv` | `d4606e189d9fd25dd3e84933bf7e69f6a940684cda9c6e78d0dbb0800aea98c6` |
| gefcom2014_wind_zone3 | 60 | 9.97 | `gefcom2014_wind_zone3_past_weather_points.csv` | `2e5843ab5767ca7c64a132976126aeb6b8c3a85c3fb4168a3e98629998101e54` |
| gefcom2014_wind_zone4 | 138 | 22.82 | `gefcom2014_wind_zone4_past_weather_points.csv` | `12358cb06f58261c9fcbe1af7ec151dc73b7b298564b25b285585e8439fe85fb` |
| gefcom2014_wind_zone5 | 138 | 23.15 | `gefcom2014_wind_zone5_past_weather_points.csv` | `7f37d4c616ba6ed79b9606bd6aa6f2e7a12bcbb3c1fd2458c783d335961b6393` |
| gefcom2014_wind_zone6 | 138 | 23.11 | `gefcom2014_wind_zone6_past_weather_points.csv` | `48d6beb61b6b8667f8946e43a0e458591773e23f2bdff014db5d4bb1e0eabae1` |
| gefcom2014_wind_zone7 | 138 | 25.21 | `gefcom2014_wind_zone7_past_weather_points.csv` | `c69ecfd31a6737588f2027d8c269fa86877e5f763607f0c7d0fdeb47350d05cb` |
| gefcom2014_wind_zone8 | 138 | 23.82 | `gefcom2014_wind_zone8_past_weather_points.csv` | `fb35668db4fe5050121ed25a77cfb206dfe9622fa23bdbb3ff4b0c3771078951` |
| gefcom2014_wind_zone9 | 138 | 23.63 | `gefcom2014_wind_zone9_past_weather_points.csv` | `720bc912bb2eaf08731f790888ea4ace5adef5e32b3fab9fe817ef570a8d1dd5` |
