"""Three pilot models on the same windows.

* ``PatchTST``: vendor PatchTST backbone, RevIN on, channel-mixing head.
* ``PatchTST-PLE-quantile``: the same network plus a zero-init quantile PLE
  residual on pre-affine RevIN coordinates.
* ``TabM``: ``tabm.TabM`` from the installed ``tabm==0.0.3`` package, with the
  same ``make`` arguments as ``run_detect.py``. The window is flattened
  ``[time, channel]`` (C-contiguous). TabM does not use RevIN.

Classification returns logits and is not passed through RevIN denorm.
Forecasting returns the cnt trajectory in the global-scaler space: the head
is trained in the cnt channel's affine RevIN space and denormalized with that
channel's lookback statistics inside the same forward.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from .config import (
    D_FF,
    D_MODEL,
    DROPOUT,
    EQUAL_BIN_MODEL,
    N_BINS,
    N_HEADS,
    N_LAYERS,
    NORM,
    PADDING_PATCH,
    PATCH_LEN,
    PATCH_STRIDE,
    PILOT_MODELS,
    REVIN_EPS,
    TABM_BLOCKS,
    TABM_D_BLOCK,
    TABM_K,
)
from .ple import QuantilePLE, instance_norm

_EXP = Path(__file__).resolve().parents[1]
_VENDOR = _EXP / "vendor"
for _p in (str(_EXP), str(_VENDOR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fast_attention import enable  # noqa: E402
from layers.PatchTST_backbone import PatchTST_backbone  # noqa: E402
from layers.RevIN import RevIN  # noqa: E402
from tabm import TabM  # noqa: E402


def denorm_channel(y_aff, mean, std, weight, bias, channel: int, eps: float = REVIN_EPS):
    """Invert vendor RevIN for one channel.

    ``y_aff`` is ``[batch, horizon]`` in affine-normalized space.
    ``mean`` and ``std`` are ``[batch, 1, channel]`` from the lookback window.
    ``weight`` and ``bias`` are ``[channel]``.
    """
    w = weight[channel]
    b = bias[channel]
    std_c = std[:, :, channel]
    mean_c = mean[:, :, channel]
    y = (y_aff - b) / (w + eps * eps)
    return y * std_c + mean_c


class RevINPatchTST(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        lookback: int,
        task: str,
        horizon: int = 1,
        dropout: float = DROPOUT,
        use_ple: bool = False,
        use_revin: bool = True,
        edges: torch.Tensor | None = None,
        cnt_index: int = 0,
    ):
        super().__init__()
        if task not in ("classification", "forecast"):
            raise ValueError(task)
        if lookback < PATCH_LEN:
            raise ValueError(f"lookback {lookback} < patch_len {PATCH_LEN}")
        self.task = task
        self.lookback = lookback
        self.n_features = n_features
        self.horizon = 1 if task == "classification" else horizon
        self.cnt_index = cnt_index
        self.use_ple = use_ple
        if not use_revin and task != "classification":
            raise ValueError("no-RevIN ablation is classification-only")
        self.use_revin = use_revin
        # RevIN initializes with constants and consumes no RNG; removing its
        # 2*C affine parameters preserves the common backbone initialization.
        self.revin = RevIN(n_features, eps=REVIN_EPS, affine=True, subtract_last=False) if use_revin else None
        self.ple = None
        if use_ple:
            if edges is None:
                raise ValueError("quantile PLE requires edges fit on training RevIN coordinates")
            self.ple = QuantilePLE(edges)
            if self.ple.n_features != n_features:
                raise ValueError("edge channels do not match n_features")
        # Same constructor call as run_detect.Model. The vendored PatchTST_backbone
        # accepts ``norm`` but does not forward it to TSTiEncoder, whose default is
        # BatchNorm1d. Both PatchTST variants keep that behavior. RevIN itself has
        # no running statistics; BatchNorm does, so predict() must call eval().
        self.net = PatchTST_backbone(
            c_in=n_features,
            context_window=lookback,
            target_window=1,
            patch_len=PATCH_LEN,
            stride=PATCH_STRIDE,
            padding_patch=PADDING_PATCH,
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_ff=D_FF,
            norm=NORM,
            dropout=dropout,
            head_dropout=0.0,
            revin=False,
            individual=False,
        )
        out_dim = 1 if task == "classification" else horizon
        self.net.head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(n_features * self.net.head_nf, out_dim),
        )
        self.n_patches = self.net.head_nf // D_MODEL
        enable(self)

    def pre_affine(self, x: torch.Tensor):
        """Per-window RevIN before the learnable affine. Also stores mean and std."""
        self._check(x)
        if self.revin is None:
            return x
        self.revin._get_statistics(x)
        z = (x - self.revin.mean) / self.revin.stdev
        return z

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[1] != self.lookback or x.shape[2] != self.n_features:
            raise ValueError(
                f"expected [batch, {self.lookback}, {self.n_features}], got {tuple(x.shape)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre_affine(x)
        if self.ple is not None:
            z = self.ple(z)
        if self.revin is not None and self.revin.affine:
            z = z * self.revin.affine_weight + self.revin.affine_bias
        # Backbone consumes [batch, channel, time] and does not run RevIN again.
        raw = self.net(z.transpose(1, 2))
        if self.task == "classification":
            return raw.reshape(x.shape[0], -1)
        y_aff = raw.reshape(x.shape[0], self.horizon)
        return denorm_channel(
            y_aff,
            self.revin.mean,
            self.revin.stdev,
            self.revin.affine_weight,
            self.revin.affine_bias,
            self.cnt_index,
            eps=self.revin.eps,
        )


class TabMWindow(nn.Module):
    """Real TabM on a flattened window. No RevIN and no PLE."""

    def __init__(self, *, n_features: int, lookback: int, d_out: int, dropout: float = DROPOUT):
        super().__init__()
        self.n_features = n_features
        self.lookback = lookback
        self.d_out = d_out
        self.net = TabM.make(
            n_num_features=n_features * lookback,
            d_out=d_out,
            n_blocks=TABM_BLOCKS,
            d_block=TABM_D_BLOCK,
            dropout=dropout,
            k=TABM_K,
        )
        if type(self.net).__name__ != "TabM" or not type(self.net).__module__.startswith("tabm"):
            raise TypeError(f"expected tabm.TabM, got {type(self.net)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.lookback or x.shape[2] != self.n_features:
            raise ValueError(
                f"expected [batch, {self.lookback}, {self.n_features}], got {tuple(x.shape)}"
            )
        # [batch, time, channel] -> [batch, time * channel], time major.
        flat = x.reshape(x.shape[0], -1)
        return self.net(flat)


def build_model(
    name: str,
    *,
    task: str,
    n_features: int,
    lookback: int,
    horizon: int = 1,
    dropout: float = DROPOUT,
    edges: torch.Tensor | None = None,
    cnt_index: int = 0,
    use_revin: bool = True,
) -> nn.Module:
    if name == EQUAL_BIN_MODEL:
        raise ValueError(
            f"{EQUAL_BIN_MODEL} stays in run_detect.py and is not part of this pilot"
        )
    if name not in PILOT_MODELS:
        raise ValueError(f"unknown model {name}; pilot models are {PILOT_MODELS}")
    d_out = 1 if task == "classification" else horizon
    if name == "TabM":
        return TabMWindow(n_features=n_features, lookback=lookback, d_out=d_out, dropout=dropout)
    return RevINPatchTST(
        n_features=n_features,
        lookback=lookback,
        task=task,
        horizon=horizon,
        dropout=dropout,
        use_ple=name == "PatchTST-PLE-quantile",
        use_revin=use_revin,
        edges=edges,
        cnt_index=cnt_index,
    )


def classification_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits on every TabM member, or on the single PatchTST logit.

    No ``pos_weight`` — same decision as the existing SKAB runs.
    """
    target = y.reshape(-1, *([1] * (pred.ndim - 1))).expand_as(pred)
    return nn.functional.binary_cross_entropy_with_logits(pred, target)


def forecast_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """MSE in global-scaler cnt units, on every TabM member or the PatchTST output."""
    if pred.ndim == 2:
        return nn.functional.mse_loss(pred, y)
    return nn.functional.mse_loss(pred, y.unsqueeze(1).expand_as(pred))


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, *, task: str, batch: int = 512) -> torch.Tensor:
    """Eval-mode predictions.

    Classification: probability in ``[0, 1]``, mean of TabM members, shape ``[N]``.
    Forecast: scaler-space cnt, mean of TabM members, shape ``[N, H]``.
    """
    model.eval()
    outs = []
    for start in range(0, x.shape[0], batch):
        raw = model(x[start : start + batch])
        if task == "classification":
            prob = raw.sigmoid()
            if prob.ndim == 3:
                prob = prob.mean(dim=1)
            outs.append(prob.reshape(prob.shape[0]))
        else:
            if raw.ndim == 3:
                raw = raw.mean(dim=1)
            outs.append(raw)
    return torch.cat(outs, dim=0)


def copy_shared_weights(src: nn.Module, dst: nn.Module) -> None:
    """Copy matching parameters and buffers. Used to compare PLE at W=0."""
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    copied = {k: v for k, v in src_state.items() if k in dst_state and dst_state[k].shape == v.shape}
    dst.load_state_dict(copied, strict=False)
