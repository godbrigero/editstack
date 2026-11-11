"""Hand-derived weight edits. Each takes the state dict `sd` (name -> tensor) and edits in place."""
import math
import torch

TMLP = "model.action_time_mlp_in.weight"   # [720, 1440]: cols 0-719 action emb, 720-1439 time emb
AIN = "model.action_in_proj.weight"        # [720, 32]
AOUT_W, AOUT_B = "model.action_out_proj.weight", "model.action_out_proj.bias"


def time_shift(sd, delta, min_period=0.004, max_period=4.0):
    """Make the network see tau-delta instead of tau, exactly: rotate each (sin_k, cos_k) column pair.
    sin(w(t-d)) = sin(wt)cos(wd) - cos(wt)sin(wd);  cos(w(t-d)) = cos(wt)cos(wd) + sin(wt)sin(wd)."""
    W = sd[TMLP].double()
    frac = torch.linspace(0.0, 1.0, 360, dtype=torch.float64)
    w = 2 * math.pi / (min_period * (max_period / min_period) ** frac)
    c, s = torch.cos(w * delta), torch.sin(w * delta)
    Ws, Wc = W[:, 720:1080].clone(), W[:, 1080:1440].clone()
    W[:, 720:1080] = Ws * c - Wc * s
    W[:, 1080:1440] = Wc * c + Ws * s
    sd[TMLP] = W.float()


def pad_zero(sd):
    """Disconnect padded action dims 6..31 (pure noise at inference, never supervised) from the input."""
    sd[AIN][:, 6:] = 0


def out_gain(sd, s, dims=slice(0, 6)):
    sd[AOUT_W][dims] *= s
    sd[AOUT_B][dims] *= s


def out_bias(sd, raw_shift, astd):
    """Shift final actions by raw_shift (raw units) via the velocity bias; final x shift = -0.1*db/gain
    for an ideal last step, so db = -raw_shift/std/0.1 (empirically calibrated by probe)."""
    for i, r in enumerate(raw_shift):
        sd[AOUT_B][i] -= float(r) / float(astd[i]) / 0.1

SPROJ_W, SPROJ_B = "model.state_proj.weight", "model.state_proj.bias"
AIN_B = "model.action_in_proj.bias"


def tau_coeffs(min_period=0.004, max_period=4.0, ridge=1e-6):
    """c (720,) with phi(tau) @ c ~= tau on [0,1] (phi = SmolVLA's sin/cos time features);
    weighted to be essentially exact at the 10 inference times 1.0..0.1."""
    frac = torch.linspace(0.0, 1.0, 360, dtype=torch.float64)
    w = 2 * math.pi / (min_period * (max_period / min_period) ** frac)
    taus_inf = torch.tensor([1.0 - 0.1 * i for i in range(10)], dtype=torch.float32).double()
    grid = torch.cat([torch.linspace(0, 1, 2001, dtype=torch.float64), taus_inf.repeat(200)])
    phi = torch.cat([torch.sin(grid[:, None] * w), torch.cos(grid[:, None] * w)], 1)
    c = torch.linalg.solve(phi.T @ phi + ridge * torch.eye(720, dtype=torch.float64), phi.T @ grid)
    p_inf = torch.cat([torch.sin(taus_inf[:, None] * w), torch.cos(taus_inf[:, None] * w)], 1)
    err = (p_inf @ c - taus_inf).abs().max().item()
    return c, err


def translate_actions(sd, raw_delta, astd, sstd, state_too=True):
    """Exact translation of the policy's action space by raw_delta (raw units, 6 dims):
    samples become y + delta. x_tau = tau*eps + (1-tau)*(y+delta), so the network must see
    x - (1-tau)*delta and output v - delta."""
    d = torch.zeros(32, dtype=torch.float64)
    d[:6] = torch.tensor([float(r) / float(s) for r, s in zip(raw_delta, astd)], dtype=torch.float64)
    A = sd[AIN].double()
    Ad = A @ d                                         # (720,)
    sd[AIN_B] = (sd[AIN_B].double() - Ad).float()      # constant part of -(1-tau)*delta
    W1 = sd[TMLP].double()
    c, err = tau_coeffs()
    W1[:, 720:] += torch.outer(W1[:, :720] @ Ad, c)    # +tau*delta part via time features
    sd[TMLP] = W1.float()
    sd[AOUT_B] = (sd[AOUT_B].double() - d).float()
    if state_too:
        ds = torch.zeros(32, dtype=torch.float64)
        ds[:6] = torch.tensor([float(r) / float(s) for r, s in zip(raw_delta, sstd)], dtype=torch.float64)
        sd[SPROJ_B] = (sd[SPROJ_B].double() - sd[SPROJ_W].double() @ ds).float()
    return err


def scale_axis(sd, dim, k, amean, astd, smean, sstd, center=None):
    """Affine rescale of one joint axis about `center` (raw units, default: action mean):
    edited policy commands y' = C + k*(y - C) and perceives state s_eff = C + (s - C)/k.
    Exact flow-matching algebra except that the network sees pan input noise of std 1/k at tau=1."""
    C = float(amean[dim]) if center is None else float(center)
    b = (1 - k) * (C - float(amean[dim])) / float(astd[dim])     # normalized action offset
    A = sd[AIN].double(); col = A[:, dim].clone() / k
    A[:, dim] = col; sd[AIN] = A.float()
    if b != 0.0:
        Ad = col * b
        sd[AIN_B] = (sd[AIN_B].double() - Ad).float()
        W1 = sd[TMLP].double(); c, _ = tau_coeffs()
        W1[:, 720:] += torch.outer(W1[:, :720] @ Ad, c); sd[TMLP] = W1.float()
    sd[AOUT_W][dim] *= k
    sd[AOUT_B][dim] = sd[AOUT_B][dim] * k - b
    Ws = sd[SPROJ_W].double(); wcol = Ws[:, dim].clone()
    sd[SPROJ_B] = (sd[SPROJ_B].double() + wcol * (C - float(smean[dim])) * (1 - 1 / k) / float(sstd[dim])).float()
    Ws[:, dim] = wcol / k; sd[SPROJ_W] = Ws.float()
