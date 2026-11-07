"""Per-feature quantile PLE in the coordinates the encoder actually reads.

Coordinate system
-----------------
PatchTST RevIN (vendor ``layers/RevIN.py``) on an input ``[batch, time, channel]``
subtracts the per-window, per-channel mean and divides by the per-window,
per-channel population standard deviation. The mean and std run over time only
(``dim=1``), with ``unbiased=False`` and ``sqrt(var + eps)``. Learnable affine
scale and bias are applied after that, and they move during training.

PLE is applied on the pre-affine normalized values. Bin edges are quantiles of
those same values, computed from training windows only. Fitting edges on raw
series, or on globally scaled series, and then encoding RevIN outputs is a
different coordinate system and is not what this module does.

The residual is the one already used by equal-bin ``PLEInput``::

    z' = z + (PLE(z) * W).sum(-1)

with ``W`` initialized at 0. Outside ``[first edge, last edge]`` every component
is clamped to ``[0, 1]`` (saturation, no linear extrapolation). A tied edge
(width <= ``WIDTH_EPS``) is a step: 0 below the edge, 1 at or above it.
That covers a constant sensor, whose RevIN output is identically 0.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import N_BINS, REVIN_EPS, WIDTH_EPS


def instance_norm(x: torch.Tensor, eps: float = REVIN_EPS):
    """Match RevIN pre-affine norm. ``x`` is ``[batch, time, channel]``.

    Statistics are detached, as in the vendor module, so the backward pass
    scales the input by ``1/std`` and does not differentiate through the
    window mean and variance.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [batch, time, channel], got {tuple(x.shape)}")
    mean = torch.mean(x, dim=1, keepdim=True).detach()
    var = torch.var(x, dim=1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps).detach()
    return (x - mean) / std, mean, std


@torch.no_grad()
def fit_revin_quantile_edges(
    x_scaled: torch.Tensor,
    n_bins: int = N_BINS,
    eps: float = REVIN_EPS,
    chunk: int = 1024,
) -> torch.Tensor:
    """Quantile edges of pre-affine RevIN(train windows).

    Parameters
    ----------
    x_scaled:
        Training input windows only, already mapped by the train-fold scaler.
        Shape ``[n_windows, lookback, n_features]``. Labels, validation, and
        test windows are not arguments of this function.
    Returns
    -------
    Tensor ``[n_features, n_bins + 1]``, float32, on CPU.
    """
    if not isinstance(x_scaled, torch.Tensor):
        x_scaled = torch.as_tensor(x_scaled, dtype=torch.float32)
    x_scaled = x_scaled.detach().to(dtype=torch.float32, device="cpu")
    if x_scaled.ndim != 3:
        raise ValueError(f"train windows must be [N, L, C], got {tuple(x_scaled.shape)}")
    if x_scaled.shape[0] < 2:
        raise ValueError("need at least 2 training windows to fit quantiles")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if not torch.isfinite(x_scaled).all():
        raise ValueError("training windows contain non-finite values")

    pieces = []
    for part in x_scaled.split(chunk, dim=0):
        z, _, _ = instance_norm(part, eps=eps)
        pieces.append(z.reshape(-1, z.shape[-1]))
    flat = torch.cat(pieces, dim=0)
    if not torch.isfinite(flat).all():
        raise ValueError("RevIN coordinates contain non-finite values")
    probs = torch.linspace(0.0, 1.0, n_bins + 1, dtype=flat.dtype)
    # torch.quantile(..., dim=0) -> [n_bins+1, C]; store [C, n_bins+1].
    edges = torch.quantile(flat, probs, dim=0).transpose(0, 1).contiguous()
    return edges.float().cpu()


def ple_encode(z: torch.Tensor, edges: torch.Tensor, width_eps: float = WIDTH_EPS) -> torch.Tensor:
    """Paper-style PLE with saturation outside the outer edges.

    ``z`` is ``[..., C]`` and ``edges`` is ``[C, n_bins+1]``.
    Output is ``[..., C, n_bins]`` in ``[0, 1]``.
    """
    if edges.ndim != 2:
        raise ValueError(f"edges must be [C, n_bins+1], got {tuple(edges.shape)}")
    if z.shape[-1] != edges.shape[0]:
        raise ValueError(
            f"channel mismatch: values have {z.shape[-1]} channels, edges have {edges.shape[0]}"
        )
    lo = edges[:, :-1]
    width = edges[:, 1:] - lo
    degen = width <= width_eps
    safe_width = torch.where(degen, torch.ones_like(width), width)
    values = z.unsqueeze(-1)
    linear = (values - lo) / safe_width
    step = (values >= lo).to(dtype=z.dtype)
    return torch.where(degen, step, linear).clamp(0.0, 1.0)


class QuantilePLE(nn.Module):
    """``z' = z + W · PLE(z)`` with frozen per-feature quantile edges."""

    def __init__(self, edges: torch.Tensor):
        super().__init__()
        edges = torch.as_tensor(edges, dtype=torch.float32).detach().cpu().clone()
        if edges.ndim != 2 or edges.shape[1] < 2:
            raise ValueError(f"edges must be [C, n_bins+1] with n_bins>=1, got {tuple(edges.shape)}")
        if not torch.isfinite(edges).all():
            raise ValueError("bin edges must be finite")
        self.register_buffer("edges", edges)
        n_features, n_edges = edges.shape
        self.W = nn.Parameter(torch.zeros(n_features, n_edges - 1))

    @property
    def n_features(self) -> int:
        return int(self.edges.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.edges.shape[1] - 1)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return ple_encode(z, self.edges)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [batch, time, channel] in pre-affine RevIN coordinates.
        return z + (self.encode(z) * self.W).sum(dim=-1)


def degenerate_edge_fraction(edges: torch.Tensor, width_eps: float = WIDTH_EPS) -> float:
    width = edges[:, 1:] - edges[:, :-1]
    return float((width <= width_eps).float().mean().item())


@torch.no_grad()
def fit_scaled_quantile_edges(x_scaled: torch.Tensor, n_bins: int = N_BINS) -> torch.Tensor:
    """TRAIN-window quantiles in global StandardScaler coordinates, no RevIN.

    No validation/test data or labels accepted. Unlike fit_revin_quantile_edges,
    these knots must only encode inputs that have NOT been instance-normalized.
    """
    x = torch.as_tensor(x_scaled, dtype=torch.float32, device="cpu").detach()
    if x.ndim != 3 or x.shape[0] < 2 or n_bins < 2:
        raise ValueError("expected at least two [N,L,C] train windows and n_bins>=2")
    if not torch.isfinite(x).all():
        raise ValueError("training windows contain non-finite values")
    probs = torch.linspace(0., 1., n_bins + 1, dtype=x.dtype)
    return torch.quantile(x.reshape(-1, x.shape[-1]), probs, dim=0).T.contiguous()
