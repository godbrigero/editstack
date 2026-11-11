"""Exact flow-matching algebra for SmolVLA action heads (numpy port of the original project's edits.py).

Action head: x (50x32 noisy actions, 6 real + 26 padding) -> action_in_proj [720,32] -> fused with sin/cos time
features in action_time_mlp_in [720, 1440] (cols 0-719 action embedding, 720-1439 time embedding) -> expert ->
action_out_proj [32, 720] -> velocity v. 10 Euler steps x <- x - 0.1 v from tau=1 to 0. Loss only on channels 0-5.
"""
import numpy as np

TMLP = "model.action_time_mlp_in.weight"
AIN, AIN_B = "model.action_in_proj.weight", "model.action_in_proj.bias"
AOUT_W, AOUT_B = "model.action_out_proj.weight", "model.action_out_proj.bias"
SPROJ_W, SPROJ_B = "model.state_proj.weight", "model.state_proj.bias"
N_REAL = 6


def stats(ctx):
    """Normalisation stats shipped with the checkpoint (raw = mean + std * normalised)."""
    post = ctx.extra("policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    pre = ctx.extra("policy_preprocessor_step_5_normalizer_processor.safetensors")
    return dict(amean=post["action.mean"].astype(np.float64), astd=post["action.std"].astype(np.float64),
                smean=pre["observation.state.mean"].astype(np.float64), sstd=pre["observation.state.std"].astype(np.float64))


def tau_coeffs(min_period=0.004, max_period=4.0, ridge=1e-6):
    """c (720,) with phi(tau) . c ~= tau on [0,1], essentially exact at the 10 inference taus 1.0..0.1."""
    frac = np.linspace(0.0, 1.0, 360)
    w = 2 * np.pi / (min_period * (max_period / min_period) ** frac)
    taus = np.array([1.0 - 0.1 * i for i in range(10)], dtype=np.float32).astype(np.float64)
    grid = np.concatenate([np.linspace(0, 1, 2001), np.tile(taus, 200)])
    phi = np.concatenate([np.sin(grid[:, None] * w), np.cos(grid[:, None] * w)], 1)
    c = np.linalg.solve(phi.T @ phi + ridge * np.eye(720), phi.T @ grid)
    p_inf = np.concatenate([np.sin(taus[:, None] * w), np.cos(taus[:, None] * w)], 1)
    return c, float(np.abs(p_inf @ c - taus).max())


def pad_zero(W):
    """Disconnect padded action channels 6..31: never supervised, but fed noise at every denoising step."""
    W[AIN][:, N_REAL:] = 0.0


def translate_actions(W, raw_delta, st, state_too=True):
    """Exact translation of the action space by raw_delta (raw joint units): samples become y + delta.
    x_tau = tau*eps + (1-tau)(y+delta): the net must see x - (1-tau) delta and output v - delta."""
    d = np.zeros(32)
    d[:N_REAL] = np.asarray(raw_delta, np.float64) / st["astd"][:N_REAL]
    Ad = W[AIN] @ d
    W[AIN_B] -= Ad
    c, err = tau_coeffs()
    W1 = W[TMLP]
    W1[:, 720:] += np.outer(W1[:, :720] @ Ad, c)
    W[AOUT_B] -= d
    if state_too:
        ds = np.zeros(32)
        ds[:N_REAL] = np.asarray(raw_delta, np.float64) / st["sstd"][:N_REAL]
        W[SPROJ_B] -= W[SPROJ_W] @ ds
    return err


def scale_axis(W, dim, k, st, center=None):
    """Affine rescale of joint `dim` about centre C (raw units, default action mean): commands y' = C + k (y - C),
    perceived state s_eff = C + (s - C)/k. Holding still is unchanged; every image-driven move is k x larger."""
    C = float(st["amean"][dim]) if center is None else float(center)
    b = (1 - k) * (C - float(st["amean"][dim])) / float(st["astd"][dim])
    A = W[AIN]
    col = A[:, dim] / k
    A[:, dim] = col
    if b != 0.0:
        Ad = col * b
        W[AIN_B] -= Ad
        c, _ = tau_coeffs()
        W[TMLP][:, 720:] += np.outer(W[TMLP][:, :720] @ Ad, c)
    W[AOUT_W][dim] *= k
    W[AOUT_B][dim] = W[AOUT_B][dim] * k - b
    Ws = W[SPROJ_W]
    wcol = Ws[:, dim].copy()
    W[SPROJ_B] += wcol * (C - float(st["smean"][dim])) * (1 - 1 / k) / float(st["sstd"][dim])
    Ws[:, dim] = wcol / k
