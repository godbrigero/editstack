"""Closed-form fitting and probing on recorded activations (no gradients, no training loop).

Typical use inside a probe script:
    X = captured inputs to a layer   [N, d_in]   (rows from real rollouts, grouped by episode/placement)
    Y = what that layer *should* output to fix the behaviour [N, d_out]
    D = fit_delta(X, Y, W_current, b_current, lam=1e-2, rank=1)    # weight delta, rank-limited
    np.save("edits/<variant>/data/delta.npy", D)                    # edit.py then does ops.add_delta(...)

Always validate with GROUPED cross-validation (by episode / placement, never by frame): frames of one episode are
near-duplicates and leak. A past camera-map fit was cross-validated by scene, not placement, and its gains were
partly placement overfit.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1e-3, intercept: bool = True):
    """argmin_B ||X B + c - Y||^2 + lam*N*||B||^2. Returns (B [d_in, d_out], c [d_out])."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    mx = X.mean(0) if intercept else np.zeros(X.shape[1])
    my = Y.mean(0) if intercept else np.zeros(Y.shape[1])
    Xc, Yc = X - mx, Y - my
    n, d = Xc.shape
    if d <= n:
        B = np.linalg.solve(Xc.T @ Xc + lam * n * np.eye(d), Xc.T @ Yc)
    else:   # dual form
        B = Xc.T @ np.linalg.solve(Xc @ Xc.T + lam * n * np.eye(n), Yc)
    return B, my - mx @ B


def truncate_rank(D: np.ndarray, r: int) -> np.ndarray:
    U, s, Vt = np.linalg.svd(np.asarray(D, np.float64), full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


def fit_delta(X: np.ndarray, Y_target: np.ndarray, W: np.ndarray, b: Optional[np.ndarray] = None,
              lam: float = 1e-3, rank: Optional[int] = None):
    """Weight delta D (same shape as W [out, in]) so that X (W + D)^T + b ~= Y_target, with ridge penalty on D.
    Returns (D, db) where db is the bias correction (zeros if b is None and you don't want one)."""
    X = np.asarray(X, np.float64)
    pred = X @ np.asarray(W, np.float64).T + (0 if b is None else np.asarray(b, np.float64))
    R = np.asarray(Y_target, np.float64) - pred
    B, c = ridge(X, R, lam=lam, intercept=b is not None)
    D = B.T
    if rank is not None:
        D = truncate_rank(D, rank)
        c = (R - X @ D.T).mean(0) if b is not None else np.zeros(R.shape[1])
    return D, c


def paired_map(X_new: np.ndarray, X_old: np.ndarray, lam: float = 1e-3, shrink_to_identity: bool = True):
    """Linear map (M, c) with M x_new + c ~= x_old, from paired samples of the same states under two conditions
    (e.g. renders from measured vs training cameras). Fold into the reader with ops.fold_input_map(W, w, M, b, c).
    With equal dims the ridge penalty shrinks M toward the IDENTITY ("no change"), not toward zero: with weak data a
    camera map should do nothing rather than erase the features (the map that worked behaved this way)."""
    X_new = np.asarray(X_new, np.float64)
    X_old = np.asarray(X_old, np.float64)
    if shrink_to_identity and X_new.shape[1] == X_old.shape[1]:
        B, c = ridge(X_new, X_old - X_new, lam=lam, intercept=True)      # residual map Delta, M = I + Delta^T
        return np.eye(X_new.shape[1]) + B.T, c
    B, c = ridge(X_new, X_old, lam=lam, intercept=True)
    return B.T, c


def _group_folds(n: int, groups: Optional[Sequence], folds: int):
    if groups is None:
        idx = np.arange(n)
        return [idx[i::folds] for i in range(folds)]
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    return [np.flatnonzero(np.isin(g, uniq[i::folds])) for i in range(min(folds, len(uniq)))]


def cv_r2(X: np.ndarray, Y: np.ndarray, groups: Optional[Sequence] = None, folds: int = 5, lam: float = 1e-3) -> Dict:
    """Grouped K-fold R^2 of a ridge read-out. Use to answer 'is quantity Y linearly present at this layer?'
    (High R^2 everywhere + failing behaviour => the problem is the read-out/action side, not perception.)"""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    pred = np.zeros_like(Y)
    for te in _group_folds(len(X), groups, folds):
        tr = np.setdiff1d(np.arange(len(X)), te)
        B, c = ridge(X[tr], Y[tr], lam=lam)
        pred[te] = X[te] @ B + c
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    return {"r2": r2.tolist(), "r2_mean": float(r2.mean()), "n": int(len(X)),
            "groups": int(len(np.unique(groups))) if groups is not None else None}


def gain(X: np.ndarray, y: np.ndarray) -> float:
    """Slope of y on x (1-D): 'how much does the output move per unit of the error it should correct?'
    Past runs found per-chunk correction gains of 0.03-0.6 where ~1 was needed."""
    x = np.asarray(X, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / max((x * x).sum(), 1e-12))
