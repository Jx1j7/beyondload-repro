"""Describe the source-file solar daily cycle without assigning a time zone."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
source = pd.read_csv(ROOT / "data/processed/gefcom2014_solar_clean.csv",
                     parse_dates=["timestamp"])
rows = []
profile_rows = []
for zone, group in source.groupby("zone"):
    power = group.power.to_numpy(float)
    hours = group.timestamp.dt.hour.to_numpy(int)
    valid = np.isfinite(power)
    mean_by_hour = pd.DataFrame({"hour": hours[valid], "power": power[valid]}).groupby("hour").power.mean()
    assert len(mean_by_hour) == 24
    angle = 2 * np.pi * mean_by_hour.index.to_numpy() / 24
    weight = mean_by_hour.to_numpy()
    centroid = (np.angle(np.sum(weight * np.exp(1j * angle))) % (2 * np.pi)) * 24 / (2 * np.pi)
    rows.append(dict(dataset=f"gefcom2014_solar_zone{zone}",
                     circular_power_centroid_hour=centroid,
                     source_midnight_mean_power=mean_by_hour.loc[0],
                     source_midnight_active_fraction=np.mean(power[valid & (hours == 0)] > .01)))
    for hour, value in mean_by_hour.items():
        profile_rows.append(dict(dataset=f"gefcom2014_solar_zone{zone}", hour=hour,
                                 mean_power=value))
out = ROOT / "experiments/competition_aligned_cpu"
pd.DataFrame(rows).to_csv(out / "solar_clock_summary.csv", index=False)
pd.DataFrame(profile_rows).to_csv(out / "solar_hourly_profile.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
