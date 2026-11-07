"""SKAB windows, identical to ``run_detect.prepare`` plus the same train-only scaler.

The label is ``anomaly`` at the right edge of the window (binary classification
at time t). ``changepoint`` is not the target. ``anomaly-free`` has no label
columns and is not in ``splits.json``; it is not used for training.

Timestamps are irregular (many gaps of about 2 seconds). The existing protocol
reindexes each file onto a 1-second grid and forward-fills at most one missing
second from the previous observed second. A window is 60 grid steps, taken
every 4 grid steps. That is not a claim that the physical sampling is a
constant 1 second. Windows never cross files, and grouped files stay in the
same side of every fold because the fold lists come from ``splits.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import run_detect  # noqa: E402

from .config import SKAB_LOOKBACK, SKAB_WINDOW_STRIDE  # noqa: E402

EXP = _EXP
FEATURES = list(run_detect.FEATURES)
SPLITS_PATH = EXP / "splits.json"


def load_splits() -> dict:
    return json.loads(SPLITS_PATH.read_text())


def windows_from_frame(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    lookback: int = SKAB_LOOKBACK,
    stride: int = SKAB_WINDOW_STRIDE,
):
    """One file -> ``(x [N, lookback, C], y [N])`` using the run_detect rule.

    Returns empty arrays when the file has no ``anomaly`` column (anomaly-free)
    or when every candidate window is non-finite.
    """
    features = list(features or FEATURES)
    if "datetime" not in frame.columns:
        raise ValueError("SKAB frame needs a datetime column")
    if "anomaly" not in frame.columns:
        empty_x = np.zeros((0, lookback, len(features)), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.float32)
        return empty_x, empty_y
    indexed = frame.copy()
    indexed.index = pd.to_datetime(indexed["datetime"])
    if not indexed.index.is_monotonic_increasing or indexed.index.has_duplicates:
        raise ValueError("datetime must be strictly increasing inside a file")
    grid = pd.date_range(indexed.index.min(), indexed.index.max(), freq="s")
    g = indexed.reindex(grid)
    # Causal fill of a single missing grid second. Labels are not filled.
    x = g[features].ffill(limit=1).to_numpy(dtype="float32")
    y = g["anomaly"].to_numpy()
    xs, ys = [], []
    for t in range(lookback - 1, len(g), stride):
        window = x[t - lookback + 1 : t + 1]
        if np.isfinite(window).all() and np.isfinite(y[t]):
            xs.append(window)
            ys.append(int(y[t]))
    if not xs:
        empty_x = np.zeros((0, lookback, len(features)), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.float32)
        return empty_x, empty_y
    return np.stack(xs).astype("float32"), np.asarray(ys, dtype="float32")


def diagnose_frame(frame: pd.DataFrame, features: list[str] | None = None) -> dict:
    """Grid, fill, and window counts for one file. Does not train anything."""
    features = list(features or FEATURES)
    out = {
        "n_raw": int(len(frame)),
        "has_anomaly": "anomaly" in frame.columns,
        "has_changepoint": "changepoint" in frame.columns,
    }
    ts = pd.to_datetime(frame["datetime"])
    dt = ts.diff().dt.total_seconds()
    out["dt_median"] = float(dt.median()) if len(dt) else None
    out["n_dt_gt1"] = int((dt > 1).sum())
    out["n_dt_le0"] = int((dt <= 0).sum())
    if "anomaly" not in frame.columns:
        out.update(
            n_grid=0,
            n_gap_cells=0,
            n_filled_cells=0,
            n_nan_after_ffill=0,
            n_candidate=0,
            n_windows=0,
            n_positive=0,
        )
        return out
    indexed = frame.copy()
    indexed.index = ts
    grid = pd.date_range(ts.min(), ts.max(), freq="s")
    g = indexed.reindex(grid)
    before = g[features]
    after = before.ffill(limit=1)
    x, y = windows_from_frame(frame, features)
    n_candidate = len(range(SKAB_LOOKBACK - 1, len(g), SKAB_WINDOW_STRIDE))
    out.update(
        n_grid=int(len(grid)),
        n_gap_cells=int(before.isna().sum().sum()),
        n_filled_cells=int((before.isna() & after.notna()).sum().sum()),
        n_nan_after_ffill=int(after.isna().sum().sum()),
        n_label_nan=int((~np.isfinite(g["anomaly"].to_numpy(dtype="float64"))).sum()),
        n_candidate=int(n_candidate),
        n_windows=int(len(y)),
        n_positive=int(y.sum()) if len(y) else 0,
    )
    return out


def load_skab_fold(fold: int) -> dict:
    """Train-only scaler and the same windows ``run_detect.run`` feeds the model.

    Returned ``x`` arrays are float32 in scaler space. ``y`` is 0/1 float32.
    Bin edges are not fit here; the caller fits them on ``train`` only.
    """
    spec = load_splits()
    folds = spec["folds"]
    if fold < 0 or fold >= len(folds):
        raise IndexError(fold)
    sp = folds[fold]
    names = sp["train"] + sp["val"] + sp["test"]
    frames, raw = run_detect.prepare(names)
    scaler = StandardScaler().fit(
        np.concatenate([frames[n][FEATURES].values for n in sp["train"]])
    )
    parts = {}
    for split in ("train", "val", "test"):
        items = [z for n in sp[split] for z in raw[n]]
        x = np.stack([z[0] for z in items])
        y = np.asarray([z[1] for z in items], dtype="float32")
        x_s = ((x - scaler.mean_) / scaler.scale_).astype("float32")
        parts[split] = {"x": x_s, "y": y, "files": list(sp[split])}
    return {
        "fold": fold,
        "parts": parts,
        "scaler_mean": np.asarray(scaler.mean_, dtype="float64"),
        "scaler_scale": np.asarray(scaler.scale_, dtype="float64"),
        "features": FEATURES,
        "task": "classification",
        "lookback": SKAB_LOOKBACK,
        "n_features": len(FEATURES),
    }


def groups_are_intact(spec: dict | None = None) -> list[dict]:
    """Each connected group lies entirely in one of train/val/test, in every fold."""
    spec = spec or load_splits()
    problems = []
    for fold in spec["folds"]:
        location = {}
        for split in ("train", "val", "test"):
            for name in fold[split]:
                if name in location:
                    problems.append({"fold": fold["fold"], "file": name, "issue": "duplicate-split"})
                location[name] = split
        for group in spec["groups"]:
            locs = [location.get(name) for name in group]
            if any(v is None for v in locs) or len(set(locs)) != 1:
                problems.append({"fold": fold["fold"], "group": group, "locs": locs})
        if any(name.startswith("anomaly-free") for name in location):
            problems.append({"fold": fold["fold"], "issue": "anomaly-free-in-fold"})
    return problems
