"""Extract the verified official GEFCom2014 supplement for this repository.

Place 1-s2.0-S0169207016000133-mmc1.zip in data/raw/ first. The archive is
not redistributed here. Only wind, solar, and load tracks are extracted.
"""
from pathlib import Path
import hashlib
import io
import zipfile

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "data/raw/1-s2.0-S0169207016000133-mmc1.zip"
DEST = BASE / "data/raw/gefcom2014_official/GEFCom2014 Data"
SHA256 = "d68d957270edd93b26a37d0f9b5e901f942abdf34c75eacbe14e417beb16e154"
TRACKS = ("W", "S", "L")


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Put the official supplement at {SOURCE}")
    with SOURCE.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError(f"Supplement SHA-256 mismatch: {digest}")
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE) as outer:
        for track in TRACKS:
            name = f"GEFCom2014 Data/GEFCom2014-{track}_V2.zip"
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                for info in inner.infolist():
                    path = Path(info.filename)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"Unsafe archive member: {info.filename}")
                target = DEST / f"GEFCom2014-{track}_V2"
                target.mkdir(exist_ok=True)
                inner.extractall(target)
                print(f"Extracted {track} to {target}")


if __name__ == "__main__":
    main()

