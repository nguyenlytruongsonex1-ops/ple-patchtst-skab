"""Measure data-processing facts. This does not train a model.

Writes ``measured_processing.json``. Counts, hashes, window accounting, and the
seasonal-naive sanity check are measurements. Parameter counts are from
constructing the networks, not from fitting them.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import N_BINS, PILOT_MODELS, SKAB_FOLDS
from .data_bike import build_bike_dataset, load_hour_frame, seasonal_naive
from .data_skab import FEATURES, diagnose_frame, groups_are_intact, load_skab_fold
from .models import build_model
from .ple import degenerate_edge_fraction, fit_revin_quantile_edges

EXP = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "measured_processing.json"
PINS = Path(__file__).resolve().parent / "data_pins.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def skab_duplicate_report() -> dict:
    """Row identity across labeled files, using the floats pandas reads.

    A sensor key is the 8-tuple of sensor values. A timestamp-sensor key also
    includes the datetime string. Conflict means one sensor key is paired with
    more than one anomaly label.
    """
    root = EXP / "data"
    frames = {}
    for folder in ("other", "valve1", "valve2"):
        for path in sorted((root / folder).glob("*.csv")):
            frames[f"{folder}/{path.name}"] = pd.read_csv(path, sep=";")
    sensor_labels = defaultdict(set)
    sensor_files = defaultdict(set)
    sensor_rows = defaultdict(int)
    ts_files = defaultdict(set)
    for name, frame in frames.items():
        values = frame[FEATURES].to_numpy()
        labels = frame["anomaly"].to_numpy()
        stamps = frame["datetime"].astype(str).to_numpy()
        for i in range(len(frame)):
            key = tuple(values[i].tolist())
            sensor_labels[key].add(float(labels[i]))
            sensor_files[key].add(name)
            sensor_rows[key] += 1
            ts_files[(stamps[i], key)].add(name)
    multi_file_keys = [k for k, files in sensor_files.items() if len(files) > 1]
    conflict_keys = [k for k in multi_file_keys if len(sensor_labels[k]) > 1]
    return {
        "definition": (
            "exact 8-sensor tuple as read by pandas; "
            "multi-file means the tuple occurs in more than one labeled file; "
            "conflict means those rows do not share a single anomaly label"
        ),
        "n_labeled_files": len(frames),
        "n_labeled_rows": int(sum(len(f) for f in frames.values())),
        "n_sensor_keys_in_multiple_files": len(multi_file_keys),
        "n_rows_in_multi_file_sensor_keys": int(sum(sensor_rows[k] for k in multi_file_keys)),
        "n_multi_file_sensor_keys_with_conflicting_labels": len(conflict_keys),
        "n_rows_in_conflicting_sensor_keys": int(sum(sensor_rows[k] for k in conflict_keys)),
        "n_timestamp_sensor_keys_in_multiple_files": int(
            sum(1 for files in ts_files.values() if len(files) > 1)
        ),
    }


def _param_counts() -> dict:
    counts = {}
    edges = torch.linspace(-2, 2, N_BINS + 1).repeat(8, 1)
    for name in PILOT_MODELS:
        model = build_model(
            name, task="classification", n_features=8, lookback=60, horizon=1, edges=edges
        )
        counts[f"skab_{name}"] = int(sum(p.numel() for p in model.parameters()))
    edges_b = torch.linspace(-2, 2, N_BINS + 1).repeat(13, 1)
    for name in PILOT_MODELS:
        model = build_model(
            name,
            task="forecast",
            n_features=13,
            lookback=168,
            horizon=24,
            edges=edges_b,
            cnt_index=0,
        )
        counts[f"bike_{name}"] = int(sum(p.numel() for p in model.parameters()))
    return counts


def measure() -> dict:
    pins = json.loads(PINS.read_text())
    expected = json.loads((EXP / "data_hashes.json").read_text())
    hash_rows = []
    mismatch = []
    for rel, digest in expected.items():
        path = EXP / rel
        if not path.exists():
            mismatch.append({"file": rel, "issue": "missing"})
            continue
        got = _sha256(path)
        hash_rows.append({"file": rel, "sha256": got, "match": got == digest})
        if got != digest:
            mismatch.append({"file": rel, "issue": "sha256"})
    free = EXP / "data" / "anomaly-free" / "anomaly-free.csv"
    free_header = free.read_text(encoding="utf-8", errors="replace").splitlines()[0] if free.exists() else None
    diagnoses = []
    if not mismatch and free.exists():
        skab_dirs = ("anomaly-free", "valve1", "valve2", "other")
        for path in sorted((EXP / "data").glob("*/*.csv")):
            if path.parent.name not in skab_dirs:
                continue
            frame = pd.read_csv(path, sep=";")
            item = diagnose_frame(frame)
            item["file"] = f"{path.parent.name}/{path.name}"
            diagnoses.append(item)
    fold_rows = []
    edge_notes = {}
    if not mismatch:
        for fold in SKAB_FOLDS:
            loaded = load_skab_fold(fold)
            edges = fit_revin_quantile_edges(torch.as_tensor(loaded["parts"]["train"]["x"]))
            edge_notes[f"fold_{fold}"] = {
                "degenerate_fraction": degenerate_edge_fraction(edges),
                "edge_min": float(edges.min()),
                "edge_max": float(edges.max()),
            }
            fold_rows.append(
                {
                    "fold": fold,
                    "n_train": int(len(loaded["parts"]["train"]["y"])),
                    "n_val": int(len(loaded["parts"]["val"]["y"])),
                    "n_test": int(len(loaded["parts"]["test"]["y"])),
                    "pos_train": float(loaded["parts"]["train"]["y"].mean()),
                    "pos_val": float(loaded["parts"]["val"]["y"].mean()),
                    "pos_test": float(loaded["parts"]["test"]["y"].mean()),
                }
            )
    bike = None
    hour = EXP / "data" / "bike" / "hour.csv"
    if hour.exists():
        got_sha = _sha256(hour)
        got_md5 = _md5(hour)
        if got_sha != pins["bike"]["hour_csv_sha256"]:
            raise SystemExit(f"hour.csv sha256 mismatch: {got_sha}")
        dataset = build_bike_dataset(load_hour_frame(hour))
        edges = fit_revin_quantile_edges(torch.as_tensor(dataset["parts"]["train"]["x"]))
        bike = {
            "hour_csv_sha256": got_sha,
            "hour_csv_md5": got_md5,
            "report": dataset["report"],
            "quantile_edges": {
                "degenerate_fraction": degenerate_edge_fraction(edges),
                "edge_min": float(edges.min()),
                "edge_max": float(edges.max()),
            },
            "seasonal_naive": [seasonal_naive(dataset, 24), seasonal_naive(dataset, 168)],
        }
    payload = {
        "kind": "processing_and_sanity_not_model_pilot",
        "stage": "pre_pilot",
        "skab_upstream_commit": pins["skab"]["commit"],
        "skab_hash_mismatches": mismatch,
        "anomaly_free_header": free_header,
        "anomaly_free_training_use": (
            "not used: the file has no anomaly/changepoint columns and is absent from splits.json"
        ),
        "group_problems": groups_are_intact(),
        "duplicates": skab_duplicate_report() if not mismatch else None,
        "skab_file_diagnostics": diagnoses,
        "skab_pilot_folds": fold_rows,
        "skab_quantile_edge_summary": edge_notes,
        "bike": bike,
        "constructed_parameter_counts": _param_counts(),
        "n_skab_files_hashed": len(hash_rows),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return payload


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def main() -> int:
    payload = measure()
    brief = {
        "wrote": str(OUT),
        "skab_mismatches": payload["skab_hash_mismatches"],
        "group_problems": payload["group_problems"],
        "skab_pilot_folds": payload["skab_pilot_folds"],
        "bike_missing_hours": None if payload["bike"] is None else payload["bike"]["report"]["n_missing_hours"],
        "bike_windows": None if payload["bike"] is None else payload["bike"]["report"]["splits"],
        "seasonal_naive": None if payload["bike"] is None else payload["bike"]["seasonal_naive"],
        "params": payload["constructed_parameter_counts"],
    }
    print(json.dumps(brief, indent=2))
    return 0
