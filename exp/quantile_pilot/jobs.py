"""Pilot job list and one-run execution.

Default plan is 18 runs: 3 models x 2 seeds x (2 SKAB folds + 1 Bike split).
The seed loop is outermost, so stopping after seed 0 leaves every model with
the same screening. Test labels are read only after the checkpoint is frozen.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import (
    BATCH_SIZE,
    BIKE_HORIZON,
    BIKE_LOOKBACK,
    EQUAL_BIN_MODEL,
    LR,
    MAX_EPOCHS,
    MAX_PILOT_RUNS,
    N_BINS,
    PATIENCE,
    PILOT_MODELS,
    PREDICT_BATCH,
    SEEDS,
    SKAB_FOLDS,
    SKAB_LOOKBACK,
    WEIGHT_DECAY,
)
from .data_bike import CNT_INDEX, build_bike_dataset, inverse_cnt, load_hour_frame
from .data_skab import load_skab_fold
from .models import build_model, predict
from .ple import fit_revin_quantile_edges
from .train import train_one

EXP = Path(__file__).resolve().parents[1]
BIKE_PATH = EXP / "data" / "bike" / "hour.csv"
REGISTRY = EXP / "quantile_pilot" / "results" / "registry.jsonl"
CKPT = EXP / "quantile_pilot" / "ckpt"


@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    seed: int
    fold: int | None = None

    @property
    def key(self) -> tuple:
        return (self.dataset, self.fold, self.model, self.seed)


def plan_jobs(
    *,
    datasets: tuple[str, ...] = ("skab", "bike"),
    models: tuple[str, ...] = PILOT_MODELS,
    seeds: tuple[int, ...] = SEEDS,
    skab_folds: tuple[int, ...] = SKAB_FOLDS,
) -> list[Job]:
    unknown = [m for m in models if m not in PILOT_MODELS]
    if unknown:
        raise ValueError(f"not in the pilot: {unknown}")
    if EQUAL_BIN_MODEL in models:
        raise ValueError(f"{EQUAL_BIN_MODEL} is not in this pilot")
    jobs: list[Job] = []
    # Seed first so a partial run is a balanced screen, not one model's full study.
    for seed in seeds:
        for dataset in datasets:
            if dataset == "skab":
                for fold in skab_folds:
                    for model in models:
                        jobs.append(Job("skab", model, seed, fold))
            elif dataset == "bike":
                for model in models:
                    jobs.append(Job("bike", model, seed, None))
            else:
                raise ValueError(dataset)
    return jobs


def assert_pilot_size(jobs: list[Job], allow_large: bool = False) -> None:
    if len(jobs) > MAX_PILOT_RUNS and not allow_large:
        raise SystemExit(
            f"{len(jobs)} runs exceed the pilot cap of {MAX_PILOT_RUNS}. "
            "Refusing to start. Pass --allow-large only for a separately approved study."
        )


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def _done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            keys.add((row["dataset"], row["fold"], row["model"], row["seed"]))
    return keys


def _classification_metrics(y, prob, val_y, val_prob) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    precision, recall, thresholds = precision_recall_curve(val_y, val_prob)
    f = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    thr = float(thresholds[int(np.argmax(f))]) if len(thresholds) else 0.5
    pred = prob >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "val_AP": float(average_precision_score(val_y, val_prob)),
        "test_AP": float(average_precision_score(y, prob)),
        "test_AUC": float(roc_auc_score(y, prob)),
        "threshold": thr,
        "test_F1": float(f1_score(y, pred)),
        "test_Precision": float(precision_score(y, pred, zero_division=0)),
        "test_Recall": float(recall_score(y, pred, zero_division=0)),
        "test_FPR": float(fp / (fp + tn)),
        "primary_metric": "AP",
    }


def _forecast_metrics(y_raw, pred_raw) -> dict:
    err = pred_raw.astype("float64") - y_raw.astype("float64")
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"test_MAE": mae, "test_RMSE": rmse, "primary_metric": "MAE"}


def _save_checkpoint(job: Job, model, snapshot: dict) -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    fold = "na" if job.fold is None else str(job.fold)
    stem = f"{job.dataset}_f{fold}_{job.model}_s{job.seed}"
    torch.save(model.state_dict(), CKPT / f"{stem}.pt")
    np.savez_compressed(CKPT / f"{stem}_preproc.npz", **snapshot)


def run_job(job: Job, *, threads: int, max_epochs: int, patience: int, batch_size: int) -> dict:
    torch.set_num_threads(threads)
    t0 = time.perf_counter()
    if job.dataset == "skab":
        data = load_skab_fold(job.fold)
        lookback = SKAB_LOOKBACK
        horizon = 1
        n_features = data["n_features"]
        cnt_index = 0
        task = "classification"
    else:
        if not BIKE_PATH.exists():
            raise FileNotFoundError(f"missing {BIKE_PATH}; run download_data.py")
        data = build_bike_dataset(load_hour_frame(BIKE_PATH))
        lookback = BIKE_LOOKBACK
        horizon = BIKE_HORIZON
        n_features = data["n_features"]
        cnt_index = CNT_INDEX
        task = "forecast"
    train = data["parts"]["train"]
    val = data["parts"]["val"]
    test = data["parts"]["test"]
    edges = None
    if job.model == "PatchTST-PLE-quantile":
        edges = fit_revin_quantile_edges(
            torch.as_tensor(train["x"], dtype=torch.float32), n_bins=N_BINS
        )
    preprocess_seconds = time.perf_counter() - t0

    def factory():
        return build_model(
            job.model,
            task=task,
            n_features=n_features,
            lookback=lookback,
            horizon=horizon,
            edges=edges,
            cnt_index=cnt_index,
        )

    t1 = time.perf_counter()
    if task == "classification":
        from sklearn.metrics import average_precision_score

        def score_from_pred(prob: np.ndarray) -> float:
            return float(average_precision_score(val["y"], prob))

        trained = train_one(
            factory,
            task=task,
            x_train=train["x"],
            y_train=train["y"],
            x_val=val["x"],
            y_val=val["y"],
            score_from_pred=score_from_pred,
            maximize=True,
            seed=job.seed,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
        )
    else:
        def inverse(pred_scaled: np.ndarray) -> np.ndarray:
            return inverse_cnt(pred_scaled, data["scaler_mean"], data["scaler_scale"])

        def score_from_pred(pred_scaled: np.ndarray) -> float:
            err = inverse(pred_scaled) - val["y_raw"]
            return float(np.mean(np.abs(err)))

        trained = train_one(
            factory,
            task=task,
            x_train=train["x"],
            y_train=train["y_scaled"],
            x_val=val["x"],
            y_val=val["y_scaled"],
            score_from_pred=score_from_pred,
            maximize=False,
            seed=job.seed,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
        )
    train_seconds = time.perf_counter() - t1
    model = trained["model"]
    t2 = time.perf_counter()
    if task == "classification":
        val_prob = predict(model, torch.as_tensor(val["x"]), task=task, batch=PREDICT_BATCH).numpy()
        test_prob = predict(model, torch.as_tensor(test["x"]), task=task, batch=PREDICT_BATCH).numpy()
        metrics = _classification_metrics(test["y"], test_prob, val["y"], val_prob)
        pred_to_save = test_prob
        y_to_save = test["y"]
    else:
        val_s = predict(model, torch.as_tensor(val["x"]), task=task, batch=PREDICT_BATCH).numpy()
        test_s = predict(model, torch.as_tensor(test["x"]), task=task, batch=PREDICT_BATCH).numpy()
        inv = lambda z: inverse_cnt(z, data["scaler_mean"], data["scaler_scale"])
        metrics = {
            "val_MAE": float(np.mean(np.abs(inv(val_s) - val["y_raw"]))),
            "val_RMSE": float(np.sqrt(np.mean((inv(val_s) - val["y_raw"]) ** 2))),
            **_forecast_metrics(test["y_raw"], inv(test_s)),
        }
        pred_to_save = inv(test_s)
        y_to_save = test["y_raw"]
    eval_seconds = time.perf_counter() - t2
    snapshot = {
        "scaler_mean": np.asarray(data["scaler_mean"], dtype="float64"),
        "scaler_scale": np.asarray(data["scaler_scale"], dtype="float64"),
        "edges": np.zeros((0,), np.float32) if edges is None else edges.numpy(),
        "cnt_index": np.int64(cnt_index),
    }
    _save_checkpoint(job, model, snapshot)
    fold_tag = "na" if job.fold is None else str(job.fold)
    pred_path = CKPT / f"{job.dataset}_f{fold_tag}_{job.model}_s{job.seed}_test.npz"
    np.savez_compressed(pred_path, y=y_to_save, pred=pred_to_save)
    row = {
        "status": "ok",
        "dataset": job.dataset,
        "fold": job.fold,
        "model": job.model,
        "seed": job.seed,
        "task": task,
        "params": int(sum(p.numel() for p in model.parameters())),
        "n_train": int(len(train["x"])),
        "n_val": int(len(val["x"])),
        "n_test": int(len(test["x"])),
        "best_epoch": trained["best_epoch"],
        "epochs_ran": trained["epochs_ran"],
        "history": trained["history"],
        "checkpoint_val": trained["best_val"],
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "n_bins": N_BINS if job.model == "PatchTST-PLE-quantile" else None,
        "cpu_threads": threads,
        "preprocess_seconds": preprocess_seconds,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "seconds": preprocess_seconds + train_seconds + eval_seconds,
        **metrics,
    }
    if task == "classification":
        row["pos_val"] = float(np.mean(val["y"]))
        row["pos_test"] = float(np.mean(test["y"]))
    return row


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Quantile PLE pilot. Does not merge or tune on test.")
    parser.add_argument("--datasets", default="skab,bike")
    parser.add_argument("--models", default=",".join(PILOT_MODELS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--skab-folds", default=",".join(str(f) for f in SKAB_FOLDS))
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    jobs = plan_jobs(
        datasets=tuple(s for s in args.datasets.split(",") if s),
        models=tuple(s for s in args.models.split(",") if s),
        seeds=tuple(int(s) for s in args.seeds.split(",") if s != ""),
        skab_folds=tuple(int(s) for s in args.skab_folds.split(",") if s != ""),
    )
    assert_pilot_size(jobs, allow_large=args.allow_large)
    print(
        json.dumps(
            {
                "plan": [j.key for j in jobs],
                "n": len(jobs),
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "note": "seed loop is outermost; test is not used for checkpointing",
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return 0
    done = _done_keys(REGISTRY)
    for job in jobs:
        if job.key in done:
            print(json.dumps({"skip": job.key}), flush=True)
            continue
        try:
            row = run_job(
                job,
                threads=args.num_threads,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )
        except Exception as exc:  # registry includes failures; then stop
            _append(
                REGISTRY,
                {
                    "status": "error",
                    "dataset": job.dataset,
                    "fold": job.fold,
                    "model": job.model,
                    "seed": job.seed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        _append(REGISTRY, row)
        print(json.dumps({k: row[k] for k in row if k != "history"}), flush=True)
    return 0
