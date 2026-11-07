"""UCI Bike Sharing hourly ``cnt`` forecast windows.

``instant`` is not a feature. ``casual`` and ``registered`` are removed because
they sum to ``cnt``. The clock is ``dteday + hr``. Missing hours stay missing:
they are not filled with zero and not imputed from the future. A window is kept
only when all ``lookback`` context hours and all ``horizon`` target hours exist
on the clock. The default lookback is 168 actual hours and the horizon is 24.

Splits are chronological on the complete hourly grid (not on the count of
observed rows). A sample is assigned by its target span: every target hour
must lie in the same split. The context may start in an earlier split, because
at decision time that past is already observed. It never includes the target
hours or any later hour. Weather and calendar enter only through that past
window. Future calendar is knowable in principle and is intentionally not fed,
so the three models see the same tensor. Future weather is not used.

The scaler is fit on finite rows whose timestamp is strictly before the train
cut, and only on the model features. Validation and test rows do not move it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import BIKE_HORIZON, BIKE_LOOKBACK, BIKE_TRAIN_FRAC, BIKE_VAL_FRAC

# Past covariates only. ``cnt`` is the lagged target, not the future label.
BIKE_FEATURES = [
    "cnt",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
CNT_INDEX = 0
EXCLUDED = ("instant", "casual", "registered", "dteday")


def timestamps_from_frame(frame: pd.DataFrame) -> pd.Series:
    if "dteday" not in frame.columns or "hr" not in frame.columns:
        raise ValueError("Bike frame needs dteday and hr")
    return pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"].astype(int), unit="h")


def load_hour_frame(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = timestamps_from_frame(frame)
    return frame


def _cuts(n_hours: int, train_frac: float, val_frac: float) -> tuple[int, int]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must leave a non-empty test tail")
    cut1 = int(n_hours * train_frac)
    cut2 = int(n_hours * (train_frac + val_frac))
    if not (0 < cut1 < cut2 < n_hours):
        raise ValueError(f"degenerate cuts {cut1}, {cut2} for {n_hours} hours")
    return cut1, cut2


def _split_of_target(start: int, horizon: int, cut1: int, cut2: int, n_hours: int) -> str | None:
    end = start + horizon - 1  # inclusive
    if end >= n_hours:
        return None
    if end < cut1:
        return "train"
    if start >= cut1 and end < cut2:
        return "val"
    if start >= cut2:
        return "test"
    return None  # target span crosses a cut


def build_bike_dataset(
    frame: pd.DataFrame,
    *,
    lookback: int = BIKE_LOOKBACK,
    horizon: int = BIKE_HORIZON,
    train_frac: float = BIKE_TRAIN_FRAC,
    val_frac: float = BIKE_VAL_FRAC,
    features: list[str] | None = None,
) -> dict:
    features = list(features or BIKE_FEATURES)
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    leaked = [c for c in ("casual", "registered", "instant") if c in features]
    if leaked:
        raise ValueError(f"refusing leak columns: {leaked}")
    if features[CNT_INDEX] != "cnt":
        raise ValueError("cnt must stay at CNT_INDEX so denorm and metrics agree")

    work = frame.copy()
    if "timestamp" not in work.columns:
        work["timestamp"] = timestamps_from_frame(work)
    work = work.sort_values("timestamp")
    if work["timestamp"].duplicated().any():
        raise ValueError("duplicate Bike timestamps")

    start = work["timestamp"].min()
    end = work["timestamp"].max()
    grid = pd.date_range(start, end, freq="h")
    indexed = work.set_index("timestamp")
    aligned = indexed.reindex(grid)
    raw = aligned[features].to_numpy(dtype="float64")
    observed = np.isfinite(raw).all(axis=1)
    n_hours = len(grid)
    cut1, cut2 = _cuts(n_hours, train_frac, val_frac)
    train_rows = observed.copy()
    train_rows[cut1:] = False
    if int(train_rows.sum()) < 2:
        raise ValueError("not enough finite train-period rows to fit the scaler")
    scaler = StandardScaler().fit(raw[train_rows])
    scale = scaler.scale_.astype("float64").copy()
    scale[scale == 0] = 1.0
    mean = scaler.mean_.astype("float64").copy()
    scaled = (raw - mean) / scale

    buckets = {k: {"x": [], "y_scaled": [], "y_raw": [], "index": []} for k in ("train", "val", "test")}
    report = {
        "n_rows_file": int(len(frame)),
        "n_grid": int(n_hours),
        "n_observed": int(observed.sum()),
        "n_missing_hours": int((~observed).sum()),
        "lookback": lookback,
        "horizon": horizon,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "cut_train_end": str(grid[cut1]),
        "cut_val_end": str(grid[cut2]),
        "grid_start": str(grid[0]),
        "grid_end": str(grid[-1]),
        "n_candidate": 0,
        "dropped_cross_split": 0,
        "features": features,
        "excluded": list(EXCLUDED),
        "cnt_index": CNT_INDEX,
        "windspeed_recorded_zeros": int((aligned["windspeed"] == 0).sum()) if "windspeed" in aligned else None,
        "hum_recorded_zeros": int((aligned["hum"] == 0).sum()) if "hum" in aligned else None,
        "cnt_min_observed": float(np.nanmin(raw[:, CNT_INDEX])) if observed.any() else None,
        "cnt_max_observed": float(np.nanmax(raw[:, CNT_INDEX])) if observed.any() else None,
        "casual_plus_registered_equals_cnt": None,
        "splits": {},
    }
    if {"casual", "registered", "cnt"}.issubset(frame.columns):
        report["casual_plus_registered_equals_cnt"] = bool(
            np.allclose(
                frame["casual"].to_numpy() + frame["registered"].to_numpy(),
                frame["cnt"].to_numpy(),
            )
        )
    for split in ("train", "val", "test"):
        report["splits"][split] = {
            "kept": 0,
            "dropped_gap": 0,
            "context_includes_earlier_split": 0,
        }

    n_candidate = 0
    for i in range(lookback, n_hours - horizon + 1):
        n_candidate += 1
        split = _split_of_target(i, horizon, cut1, cut2, n_hours)
        if split is None:
            report["dropped_cross_split"] += 1
            continue
        ctx = scaled[i - lookback : i]
        fut = raw[i : i + horizon, CNT_INDEX]
        if not (np.isfinite(ctx).all() and np.isfinite(fut).all()):
            report["splits"][split]["dropped_gap"] += 1
            continue
        y_s = ((fut - mean[CNT_INDEX]) / scale[CNT_INDEX]).astype("float32")
        buckets[split]["x"].append(ctx.astype("float32"))
        buckets[split]["y_scaled"].append(y_s)
        buckets[split]["y_raw"].append(fut.astype("float32"))
        buckets[split]["index"].append(i)
        report["splits"][split]["kept"] += 1
        cut = {"train": 0, "val": cut1, "test": cut2}[split]
        if i - lookback < cut:
            report["splits"][split]["context_includes_earlier_split"] += 1
    report["n_candidate"] = int(n_candidate)
    kept_total = sum(report["splits"][s]["kept"] for s in ("train", "val", "test"))
    gap_total = sum(report["splits"][s]["dropped_gap"] for s in ("train", "val", "test"))
    report["n_kept"] = int(kept_total)
    report["n_dropped_gap"] = int(gap_total)
    if kept_total + gap_total + report["dropped_cross_split"] != n_candidate:
        raise RuntimeError("bike window accounting does not add up")

    parts = {}
    for split in ("train", "val", "test"):
        if buckets[split]["x"]:
            x = np.stack(buckets[split]["x"])
            y_s = np.stack(buckets[split]["y_scaled"])
            y_r = np.stack(buckets[split]["y_raw"])
            index = np.asarray(buckets[split]["index"], dtype=np.int64)
        else:
            x = np.zeros((0, lookback, len(features)), np.float32)
            y_s = np.zeros((0, horizon), np.float32)
            y_r = np.zeros((0, horizon), np.float32)
            index = np.zeros((0,), np.int64)
        parts[split] = {"x": x, "y_scaled": y_s, "y_raw": y_r, "target_index": index}

    return {
        "parts": parts,
        "scaler_mean": mean,
        "scaler_scale": scale,
        "raw": raw,
        "scaled": scaled.astype("float64"),
        "observed": observed,
        "grid": grid,
        "cut1": cut1,
        "cut2": cut2,
        "features": features,
        "cnt_index": CNT_INDEX,
        "lookback": lookback,
        "horizon": horizon,
        "n_features": len(features),
        "task": "forecast",
        "report": report,
    }


def inverse_cnt(pred_scaled: np.ndarray, scaler_mean: np.ndarray, scaler_scale: np.ndarray) -> np.ndarray:
    return pred_scaled * scaler_scale[CNT_INDEX] + scaler_mean[CNT_INDEX]


def seasonal_naive(dataset: dict, period: int, split: str = "test") -> dict:
    """Repeat the cnt value from ``period`` hours earlier. Not a trained model.

    For period 24 and 168 and the default horizon 24, the source hour lies
    inside the kept context, so every kept window has a finite naive forecast.
    """
    if period < 1 or period > dataset["lookback"]:
        raise ValueError("seasonal period must lie inside the kept context")
    part = dataset["parts"][split]
    raw = dataset["raw"]
    idx = part["target_index"]
    horizon = dataset["horizon"]
    y = part["y_raw"]
    if len(idx) == 0:
        return {"period": period, "split": split, "n": 0, "MAE": None, "RMSE": None}
    if np.any(idx < period):
        raise RuntimeError("a kept window starts before the seasonal lag")
    pred = np.stack([raw[i - period : i - period + horizon, CNT_INDEX] for i in idx]).astype("float64")
    if not np.isfinite(pred).all():
        raise RuntimeError(f"seasonal naive period {period} hit a missing hour on a kept window")
    err = pred - y.astype("float64")
    return {
        "period": period,
        "split": split,
        "n": int(len(idx)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
    }
