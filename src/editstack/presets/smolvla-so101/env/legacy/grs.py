"""grasp agent: gripper-GATED roll servo, an exact SwiGLU hinge circuit at action-expert layer 15.
   dv_roll = q * a(x) * ramp(g(x); g0, g1)
     x     = L15 MLP input (post-attention RMSNorm output, 720-d), per action token, every Euler step
     a(x)  = wa.x  : linear readout of the jaw-vs-cube relative angle (deg), fit on open-gripper hover scenes with the
                     x_t-Jacobian subspace projected out (no denoising feedback) and state-roll invariance
     g(x)  = wg.x  : linear readout of the gripper STATE (units)  ->  ramp = 0 below g0, 1 above g1
   Two neurons j1, j2 (gate = k(g - g0*one) / k(g - g1*one), up = a):  silu(k(g-g0))/k * a - silu(k(g-g1))/k * a
   = a * W * ramp (soft hinges), W = g1 - g0.  Down columns = residual direction that changes ONLY the roll velocity
   (pinv through lm_expert.norm.weight and action_out_proj.weight[:6]); coefficient q/W.
   Neurons default 779, 1893 (least-used L15 units, not used by approach/circuit circuits)."""
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
EXP = 'model.vlm_with_expert.lm_expert'
RMSF = 3.3


def joint_direction(sd, dim, rmsf=RMSF):
    dv = np.zeros(6); dv[dim] = 1.0
    Wout = sd['model.action_out_proj.weight'].double().numpy(); gam = sd[f'{EXP}.norm.weight'].double().numpy()
    return rmsf * np.linalg.pinv(Wout[:6] * gam[None, :]) @ dv


def _write(sd, L, rows):
    pre = f'{EXP}.layers.{L}.mlp.'
    Gw, Uw, Dw = (sd[pre + k + '_proj.weight'].float() for k in ('gate', 'up', 'down'))
    for j, g, u, dcol in rows:
        Gw[j] = torch.tensor(g, dtype=torch.float32); Uw[j] = torch.tensor(u, dtype=torch.float32)
        Dw[:, j] = torch.tensor(dcol, dtype=torch.float32)
    for k, W in zip(('gate', 'up', 'down'), (Gw, Uw, Dw)):
        sd[pre + k + '_proj.weight'] = W.to(Gw.dtype if False else sd[pre + k + '_proj.weight'].dtype)


def gated_roll(sd, gain, readout=str(HERE / 'runs/grs_readout.npz'), g0=75.0, g1=88.0, k=None, L=15, neurons=(779, 1893), key='a'):
    """dv_roll (normalized velocity) = -gain * a_deg * ramp(gripper state).  A velocity change applied at all Euler steps
    moves the chunk by -dv, so the roll command moves by +gain*ASTD[4]*a per chunk (gain ~0.015 -> ~half the error)."""
    r = np.load(readout); wa, wg, one = r[key], r['g'], r['one']
    if k is None: k = 8.0 / (g1 - g0)
    D = joint_direction(sd, 4); W = g1 - g0; q = -gain
    rows = [(neurons[0], k * (wg - g0 * one), wa, D * q / W), (neurons[1], k * (wg - g1 * one), wa, -D * q / W)]
    # silu(k z) = k * h(z): gate already includes k, so divide the down column by k
    rows = [(j, g, u, d / k) for j, g, u, d in rows]
    _write(sd, L, rows)
    return dict(neurons=neurons, g0=g0, g1=g1, k=k)


def gated_roll2(sd, gain, readout=str(HERE / 'data/grs_readout_v2.npz'), beta=8.5, gc=85.0, hc=4.5, z1=8.0, k=None, L=15,
                neurons=(779, 1893)):
    """AND-like gate on one combined variable  z = (g - gc) + beta*(h - hc)   (g gripper-state units, h tip height cm):
         on  (ramp 1) for z > z1 : jaws open AND tip >= ~hc  (hover, before the final descent)
         off (ramp 0) for z < 0  : descending below ~3 cm, closing/closed, lifting, and the REST pose (g~39, h~8.5 -> z~-12)
       dv_roll = -gain * a_deg * ramp(z).  Same two-hinge construction as gated_roll."""
    r = np.load(readout); wa, wg, wh, one = r['a'], r['g'], r['h'], r['one']
    wz = wg + beta * wh - (gc + beta * hc) * one
    if k is None: k = 8.0 / z1
    D = joint_direction(sd, 4); q = -gain
    rows = [(neurons[0], k * wz, wa, D * q / z1 / k), (neurons[1], k * (wz - z1 * one), wa, -D * q / z1 / k)]
    _write(sd, L, rows)
    return dict(neurons=neurons, beta=beta, gc=gc, hc=hc, z1=z1, k=k)
