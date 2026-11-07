"""Download official SKAB and UCI Bike Sharing files and check pinned hashes.

Raw CSV files stay under exp/data/ and are gitignored. A hash mismatch exits
non-zero and does not silently switch to another source.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
PINS = json.loads((Path(__file__).resolve().parent / "data_pins.json").read_text())
MANIFEST = json.loads((EXP / "data_hashes.json").read_text())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def download_skab() -> None:
    blob = _get(PINS["skab"]["tarball"])
    archive = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
    found = {}
    for member in archive.getmembers():
        if not member.name.endswith(".csv") or "/data/" not in member.name:
            continue
        rel = member.name.split("/data/", 1)[1]
        found["data/" + rel] = archive.extractfile(member).read()
    missing = [key for key in MANIFEST if key not in found]
    if missing:
        raise SystemExit(f"SKAB tarball is missing {missing}")
    bad = [key for key, digest in MANIFEST.items() if _sha256(found[key]) != digest]
    if bad:
        raise SystemExit(f"SKAB sha256 mismatch for {bad}. Refusing to write.")
    for key, payload in found.items():
        path = EXP / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    print(json.dumps({"skab_files": len(found), "commit": PINS["skab"]["commit"]}))


def download_bike() -> None:
    blob = _get(PINS["bike"]["zip_url"])
    if _sha256(blob) != PINS["bike"]["zip_sha256"]:
        raise SystemExit("Bike zip sha256 mismatch. Refusing to extract.")
    archive = zipfile.ZipFile(io.BytesIO(blob))
    hour = archive.read("hour.csv")
    if _sha256(hour) != PINS["bike"]["hour_csv_sha256"]:
        raise SystemExit("hour.csv sha256 mismatch. Refusing to write.")
    path = EXP / "data" / "bike" / "hour.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hour)
    print(json.dumps({"bike_hour_sha256": _sha256(hour), "bytes": len(hour)}))


def main() -> int:
    download_skab()
    download_bike()
    return 0


if __name__ == "__main__":
    sys.exit(main())
