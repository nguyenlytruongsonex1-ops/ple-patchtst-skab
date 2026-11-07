"""One seeded training run. The test split is not an argument."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
from torch import nn

from .config import BATCH_SIZE, GRAD_CLIP, LR, MAX_EPOCHS, PATIENCE, PREDICT_BATCH, WEIGHT_DECAY
from .models import classification_loss, forecast_loss, predict


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def iter_batches(order: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    """Split a permutation into batches.

    The vendored encoder is BatchNorm1d. A training batch of size 1 raises.
    When the last batch would have one row, it is merged into the previous
    batch. The same rule is used for every model.
    """
    batches = list(order.split(batch_size))
    if len(batches) >= 2 and int(batches[-1].numel()) == 1:
        batches[-2] = torch.cat([batches[-2], batches[-1]])
        del batches[-1]
    return batches


def train_one(
    model_factory: Callable[[], nn.Module],
    *,
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    score_from_pred: Callable[[np.ndarray], float],
    maximize: bool,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    grad_clip: float = GRAD_CLIP,
) -> dict:
    """Fit on train. ``score_from_pred`` is the only validation signal.

    The callable receives eval-mode predictions on ``x_val`` and must not read
    the test split. Classification passes probabilities. Forecasting passes
    scaler-space cnt; the caller inverse-scales before computing MAE.
    """
    if task not in ("classification", "forecast"):
        raise ValueError(task)
    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError("empty train or val split")
    # Edges and the scaler are fit before this call. Seeding here only affects
    # parameter initialization and the batch permutation.
    seed_all(seed)
    model = model_factory()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed + 100)
    xtr = torch.as_tensor(x_train, dtype=torch.float32)
    ytr = torch.as_tensor(y_train, dtype=torch.float32)
    xv = torch.as_tensor(x_val, dtype=torch.float32)
    loss_fn = classification_loss if task == "classification" else forecast_loss

    best_score = None
    best_state = None
    best_epoch = 0
    wait = 0
    history = []
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(xtr), generator=gen)
        for idx in iter_batches(order, batch_size):
            pred = model(xtr[idx])
            loss = loss_fn(pred, ytr[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        pred_v = predict(model, xv, task=task, batch=PREDICT_BATCH).numpy()
        score = float(score_from_pred(pred_v))
        if not np.isfinite(score):
            raise RuntimeError(f"non-finite validation score at epoch {epoch}")
        history.append(score)
        improved = best_score is None or (score > best_score if maximize else score < best_score)
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "best_val": best_score,
        "history": history,
    }
