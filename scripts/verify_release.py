"""Read-only integrity check of the archived comparison release."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import math

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "experiments/competition_aligned_cpu"
EXPECTED_ORIGINS = 1635
EXPECTED_RESULTS_HASH = "de787fa9079cab02598904398344166142d832ac734ec918e0435999c275dea8"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert sha(BASE / "experiments/results.csv") == EXPECTED_RESULTS_HASH
    origins = defaultdict(set)
    with (ROOT / "candidate_origins.csv").open() as f:
        for row in csv.DictReader(f):
            if row["eligible"] == "True":
                origins[row["dataset"]].add(int(row["origin_index"]))
    assert len(origins) == 13 and sum(map(len, origins.values())) == EXPECTED_ORIGINS
    assert sum(len(v) for k, v in origins.items() if "_wind_" in k) == 1164
    assert sum(len(v) for k, v in origins.items() if "_solar_" in k) == 471
    actuals = {}
    for family, expected_files in (("chronos2", 26), ("lightgbm", 130)):
        directory = ROOT / family
        files = sorted(directory.glob("*_points.csv"))
        assert len(files) == expected_files, (family, len(files))
        manifest = (directory / "run_manifest.md").read_text()
        for file in files:
            assert f"`{file.name}` | `{sha(file)}`" in manifest, file.name
            dataset = "_".join(file.stem.split("_")[:3])
            assert dataset in origins
            observed = defaultdict(set)
            with file.open() as f:
                for row in csv.DictReader(f):
                    origin = int(row["origin_index"])
                    step = int(row["horizon_step"])
                    assert origin in origins[dataset] and 1 <= step <= 24
                    assert step not in observed[origin]
                    observed[origin].add(step)
                    actual = float(row["actual"])
                    prediction = float(row["prediction"])
                    assert math.isfinite(actual) and math.isfinite(prediction)
                    key = (dataset, origin, step)
                    value = (row["target_timestamp"], actual)
                    assert key not in actuals or actuals[key] == value
                    actuals[key] = value
            assert set(observed) == origins[dataset]
            assert all(len(steps) == 24 for steps in observed.values())
    assert len(actuals) == EXPECTED_ORIGINS * 24
    past_dir = ROOT / "chronos2_past_weather"
    past_files = sorted(past_dir.glob("*_past_weather_points.csv"))
    assert len(past_files) == 13, len(past_files)
    past_manifest = (past_dir / "run_manifest.md").read_text()
    for file in past_files:
        assert f"`{file.name}` | `{sha(file)}`" in past_manifest, file.name
        dataset = file.name.removesuffix("_past_weather_points.csv")
        assert dataset in origins
        seen = defaultdict(set)
        with file.open() as f:
            for row in csv.DictReader(f):
                origin = int(row["origin_index"])
                step = int(row["horizon_step"])
                assert origin in origins[dataset] and 1 <= step <= 24
                assert step not in seen[origin]
                seen[origin].add(step)
                assert (row["target_timestamp"], float(row["actual"])) == actuals[(dataset, origin, step)]
                assert math.isfinite(float(row["prediction"]))
        assert set(seen) == origins[dataset]
        assert all(len(steps) == 24 for steps in seen.values())
    print("PASS: 13 sites, 1,635 common origins, 156 primary + 13 past-weather prediction files, 39,240 shared targets")


if __name__ == "__main__":
    main()
