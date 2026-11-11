"""Exact weight-edit operations. All work on the float64 working copies in `W` (see variant.Weights).

Conventions: a linear layer computes y = x @ W.T + b with W shaped [out, in] (PyTorch convention).
If your checkpoint stores [in, out] (e.g. some JAX/TF exports), transpose your thinking, not the tensor.

Three families (in order of preference, from what worked in past runs):
  1. EXACT ALGEBRA  - provably function-preserving re-parameterisations, plus a deliberate, closed-form change
                      (disconnect never-trained inputs, shift/scale an axis, fold a linear map into a layer).
  2. CLOSED-FORM FITS - least-squares / ridge on recorded activations (see editstack.fit), written with
                      lowrank()/rank1() and limited rank.
  3. GATED CIRCUITS - new mechanisms in spare (least-used) units, built so they output exactly zero in the
                      situations the model already handles (claim the units with ctx.claim).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import claims as claims_mod

# ------------------------------------------------------------------------------------------- 1. exact algebra


def zero_cols(W, name: str, cols) -> None:
    """Disconnect inputs: e.g. padded channels that were never supervised but still receive noise at inference."""
    W[name][:, cols] = 0.0


def zero_rows(W, name: str, rows, bias: Optional[str] = None) -> None:
    W[name][rows, :] = 0.0
    if bias:
        W[bias][rows] = 0.0


def scale_out(W, weight: str, k, rows=None, bias: Optional[str] = None) -> None:
    """Multiply output channels by k (behaviour-changing)."""
    sl = slice(None) if rows is None else rows
    W[weight][sl] *= k
    if bias:
        W[bias][sl] *= k


def compensate_input_shift(W, weight: str, bias: str, cols, shift) -> None:
    """The consumer y = A x + b will now receive x[cols] + shift; keep y unchanged: b -= A[:, cols] @ shift. Exact."""
    A = W[weight]
    W[bias] -= A[:, cols] @ np.atleast_1d(np.asarray(shift, dtype=np.float64))


def compensate_input_scale(W, weight: str, cols, k) -> None:
    """The consumer will receive x[cols] * k; keep its output unchanged by dividing those columns by k. Exact."""
    W[weight][:, cols] /= k


def rescale_channel(W, producer_w: str, producer_b: Optional[str], consumer_w: str, idx, k) -> None:
    """Producer channel idx *k, consumer column idx /k. Function-preserving iff this consumer is the channel's only
    (linear) reader. Use it to *verify* your understanding (output must be bit-close to the original), then drop
    one half on purpose to get a behaviour change (e.g. 'scale_axis': gain on commands, not on perception)."""
    scale_out(W, producer_w, k, rows=idx, bias=producer_b)
    compensate_input_scale(W, consumer_w, idx, k)


def shift_bias(W, bias: str, idx, delta) -> None:
    W[bias][idx] += delta


def fold_input_map(W, weight: str, M: np.ndarray, bias: Optional[str] = None, c: Optional[np.ndarray] = None) -> None:
    """Make the layer see M x + c instead of x:  W <- W M,  b <- b + W c.
    (Used to map features under new/measured cameras back to the features the model was trained on.)"""
    Wt = W[weight]
    if c is not None:
        if not bias:
            raise ValueError("fold_input_map with an offset needs the bias name")
        W[bias] += Wt @ c
    W[weight] = Wt @ M


def fold_output_map(W, weight: str, bias: Optional[str], M: np.ndarray, c: Optional[np.ndarray] = None) -> None:
    """Make the layer emit M y + c instead of y:  W <- M W,  b <- M b + c."""
    W[weight] = M @ W[weight]
    if bias:
        W[bias] = M @ W[bias] + (0 if c is None else c)
    elif c is not None:
        raise ValueError("offset needs a bias")


# ------------------------------------------------------------------------------------------- 2. low-rank updates


def rank1(W, name: str, u: np.ndarray, v: np.ndarray, alpha: float = 1.0) -> None:
    """W += alpha * u v^T  (u: [out], v: [in])."""
    W[name] += alpha * np.outer(u, v)


def lowrank(W, name: str, U: np.ndarray, V: np.ndarray) -> None:
    """W += U @ V^T  (U: [out, r], V: [in, r])."""
    W[name] += U @ V.T


def add_delta(W, name: str, D: np.ndarray, max_rank: Optional[int] = None) -> None:
    """W += D, optionally truncated to max_rank (keep edits low-rank: they generalise better and are auditable)."""
    if max_rank is not None:
        from .fit import truncate_rank
        D = truncate_rank(D, max_rank)
    W[name] += D


# ------------------------------------------------------------------------------------------- 3. circuits in spare units


def unit_usage(W, tensors: Sequence[tuple], axis_size: int) -> np.ndarray:
    """Weight-norm proxy for how much each hidden unit is used. tensors: [(name, axis), ...] all indexing the same
    units (e.g. gate rows (axis 0), up rows (axis 0), down cols (axis 1)). Prefer activation statistics from real
    rollouts when you have them (pass them to free_units(score=...))."""
    s = np.zeros(axis_size)
    for name, axis in tensors:
        a = W.original(name) if hasattr(W, "original") else W[name]
        a = np.asarray(a, dtype=np.float64)
        s += np.linalg.norm(a, axis=1 - axis) if a.ndim == 2 else np.abs(a)
    return s


def free_units(stack, W, tensors: Sequence[tuple], k: int, score: Optional[np.ndarray] = None) -> list:
    """k least-used, unclaimed units (claims checked on every listed tensor/axis)."""
    first, axis = tensors[0]
    n = np.asarray(W.original(first)).shape[axis]
    s = unit_usage(W, tensors, n) if score is None else np.asarray(score, dtype=np.float64)
    taken = set()
    for name, ax in tensors:
        taken |= claims_mod.claimed_indices(stack, name, ax)
    order = [int(i) for i in np.argsort(s) if int(i) not in taken]
    return order[:k]


def swiglu_unit(W, ctx, gate: str, up: str, down: str, idx: int, g_in: np.ndarray, u_in: np.ndarray,
                d_out: np.ndarray, purpose: str) -> None:
    """Overwrite hidden unit idx of a SwiGLU MLP (h = silu(gate x) * (up x); y = down h) and claim it."""
    W[gate][idx, :] = g_in
    W[up][idx, :] = u_in
    W[down][:, idx] = d_out
    for name, ax in ((gate, 0), (up, 0), (down, 1)):
        ctx.claim(name, ax, [idx], purpose)


def exact_product_pair(W, ctx, gate: str, up: str, down: str, pair: Sequence[int], a: np.ndarray, b: np.ndarray,
                       out: np.ndarray, purpose: str) -> None:
    """Two SwiGLU units that add (a.x)*(b.x)*out to the residual EXACTLY, since silu(z) - silu(-z) = z:
        unit1: gate=+a, up=+b ; unit2: gate=-a, up=-b  ->  silu(a.x)(b.x) + silu(-a.x)(-b.x) = (a.x)(b.x).
    Make one factor a quantity that is exactly zero when no correction is wanted (e.g. planned - current state),
    so the circuit is silent in-distribution. x is the MLP input (after its norm)."""
    i, j = pair
    swiglu_unit(W, ctx, gate, up, down, i, a, b, out, purpose + " (+)")
    swiglu_unit(W, ctx, gate, up, down, j, -a, -b, out, purpose + " (-)")
