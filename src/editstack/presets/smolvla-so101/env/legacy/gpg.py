"""agent 'approach': exact SwiGLU neuron circuits at action-expert layer 15 that change the PAN output only.

All circuits add a residual vector at the L15 MLP output whose effect, through the final RMSNorm (rms ~ RMSF) and
action_out_proj, is a change of the pan velocity only (pinv solve: exactly 0 on lift/elbow/wrist-flex/roll/gripper).
A velocity change dv applied at all 10 Euler steps moves the chunk by -dv (normalized), i.e. pan by -dv*ASTD[0] units.

Building block (from agents/circuit): two neurons with gate = +-a*(w1.x), up = +-b*(w2.x) give EXACTLY
    silu(a w1.x)*b w2.x + silu(-a w1.x)*(-b w2.x) = a*b*(w1.x)*(w2.x)      (silu(z) - silu(-z) = z)
and one neuron with gate = k*(w1.x - t), up = w2.x gives ~ k*relu(w1.x - t)*(w2.x)  (soft hinge, silu dip <= 0.28/k).

gated_gain(): dP = -c * d * (1 - rho)   with d = planned pan - state pan (readout 'd', true units / U),
    rho = ramp of a gating variable g between g0 and g1 (both signs: |g|), built from 4 hinge neurons times d.
    -> inside the band the pan servo/aim gain is (1-c) x that of pan x1.7 (c = 1 - 1/1.7 gives the export's gain),
       outside the band it is untouched (pan x1.7). Proportional to d, so exactly 0 whenever the arm holds (no drift
       beyond the d-readout bias)."""
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
EXP = 'model.vlm_with_expert.lm_expert'
RMSF = 3.3
U = 52.09   # pan units per rad

def pan_direction(sd, astd, rmsf=RMSF):
    """Residual vector r (L15 output) such that the pan velocity changes by +1 (normalized) and other dims by 0."""
    dv = np.zeros(6); dv[0] = 1.0
    Wout = sd['model.action_out_proj.weight'].double().numpy(); gam = sd[f'{EXP}.norm.weight'].double().numpy()
    return rmsf * np.linalg.pinv(Wout[:6] * gam[None, :]) @ dv

def _write(sd, L, rows):
    pre = f'{EXP}.layers.{L}.mlp.'
    Gw, Uw, Dw = (sd[pre + k + '_proj.weight'].float() for k in ('gate', 'up', 'down'))
    for j, g, u, dcol in rows:
        Gw[j] = torch.tensor(g, dtype=torch.float32); Uw[j] = torch.tensor(u, dtype=torch.float32)
        Dw[:, j] = torch.tensor(dcol, dtype=torch.float32)
    for k, W in zip(('gate', 'up', 'down'), (Gw, Uw, Dw)):
        sd[pre + k + '_proj.weight'] = W.to(torch.bfloat16)

def gated_gain(sd, astd, c=0.41, g0=0.075, g1=0.09, k=None, gate='xc', G=1.0, L=15, a=1.0, b=1.0,
               readout=HERE / 'data/readout_L15.npz', neurons=None, rmsf=RMSF):
    """dP[units] = -G*c*d*(1-rho(|g|)).  d readout in units of pan units ('d'), g readout ('xc' cube x in m, or 'p' etc)."""
    r = np.load(readout); wd, one, wg = r['d'], r['one'], r[gate]
    if neurons is None: neurons = [int(j) for j in r['neurons'][:6]]
    if k is None: k = 8.0 / (g1 - g0)          # hinge sharpness: transition ~ 1/8 of the ramp width
    D = pan_direction(sd, astd, rmsf)          # +1 normalized pan velocity
    s = G * c / float(astd[0])                 # want dv = +c*G*d*(1-rho)/ASTD  ->  dP = -c*G*d*(1-rho)
    W = g1 - g0
    rows = [(neurons[0], a * wd, b * one, D * s / (a * b)),          # + c d   (exact pair)
            (neurons[1], -a * wd, -b * one, D * s / (a * b))]
    # - c d rho(|g|):  rho = [h(g-g0) - h(g-g1) + h(-g-g0) - h(-g-g1)] / W,   h(z) = silu(k z)/k
    for j, sg, t, sgn in ((neurons[2], 1, g0, -1), (neurons[3], 1, g1, +1), (neurons[4], -1, g0, -1), (neurons[5], -1, g1, +1)):
        rows.append((j, k * (sg * wg - t * one), wd, D * s * sgn / (k * W)))
    _write(sd, L, rows)
    return dict(neurons=neurons, c=c, g0=g0, g1=g1, k=k)

def softclip(sd, astd, c=0.41, phi0=22.0, k=0.5, G=1.0, L=15, a=1.0, b=1.0, center=0.0,
             readout=HERE / 'data/readout_L15.npz', neurons=None, rmsf=RMSF, ret=None):
    """Telescoping nonlinear pan gain:  dP = G(p) - G(s)  (true pan units),
         G(phi) = -c * sat(phi - center),  sat(u) = u - h(u - phi0) + h(-u - phi0),  h(z) = silu(k z)/k  (soft relu)
       -> |phi| < phi0: slope (1-c) on every pan move (c = 1-1/1.7: the export's gain); |phi| > phi0: slope 1 (pan x1.7).
       Optional ret=(phi1, m): extra +c*m*[h(u-phi1) - h(-u-phi1)] (slope rises to 1+c*m... beyond phi1; m=1 -> back to 1).
       Exactly 0 when the arm holds (p == s); p = s + d from the 's' and 'd' readouts, so p - s == d.
       Six neurons (eight with ret)."""
    r = np.load(readout); wd, ws, one = r['d'], r['s'], r['one']
    ws = ws - center * one; wp = ws + wd
    nn = 6 if ret is None else 10
    if neurons is None: neurons = [int(j) for j in r['neurons'][:nn]]
    D = pan_direction(sd, astd, rmsf); s = G / float(astd[0])        # coefficient q on D gives dP = -q*ASTD/G...: dv=+q -> dP=-q*ASTD
    # we need dP = G(p)-G(s);  dv = -(G(p)-G(s))/ASTD  ->  coefficient = -(G(p)-G(s)) * s
    rows = [(neurons[0], a * wd, b * one, D * s * c / (a * b)),       # -(-c d) = +c d
            (neurons[1], -a * wd, -b * one, D * s * c / (a * b))]
    # G(p)-G(s) hinge part: -c*[-(h(p-phi0)-h(s-phi0)) + (h(-p-phi0)-h(-s-phi0))]  -> coefficient = -that*s
    #   = -c*s*[h(p-phi0) - h(s-phi0) - h(-p-phi0) + h(-s-phi0)]   ; neuron silu(k z)*1 = k h(z)
    hs = [(wp, phi0, -1), (ws, phi0, +1), (-wp, phi0, +1), (-ws, phi0, -1)]
    if ret is not None:
        phi1, m = ret   # +c*m*[h(u-phi1)-h(-u-phi1)] in G  -> coefficient -(.)*s
        hs += [(wp, phi1, -m), (ws, phi1, +m), (-wp, phi1, +m), (-ws, phi1, -m)]
    for j, (w, t, sgn) in zip(neurons[2:], hs):
        rows.append((j, k * (w - t * one), b * one, D * s * c * sgn / (k * b)))
    _write(sd, L, rows)
    return dict(neurons=neurons, c=c, phi0=phi0, k=k)

def boost(sd, astd, cb=0.7, g0=0.065, g1=0.095, k=None, gate='xc', G=1.0, L=15, b=1.0,
          readout=HERE / 'data/readout_L15.npz', neurons=None, rmsf=RMSF, nofs=0):
    """Out-of-band pan gain boost on the EXPORT base:  dP = +cb * d * rho(|g|)   (4 hinge neurons).
       cb = 0.7 -> every pan move x1.7 when the cube is outside the band, x1 inside. 0 when holding (d = 0)."""
    r = np.load(readout); wd, one, wg = r['d'], r['one'], r[gate]
    if neurons is None: neurons = [int(j) for j in r['neurons'][nofs:nofs + 4]]
    if k is None: k = 8.0 / (g1 - g0)
    D = pan_direction(sd, astd, rmsf); s = G * cb / float(astd[0]); W = g1 - g0
    rows = []   # coefficient q = -dP/ASTD = -cb*s'*d*rho ; rho = [h(g-g0)-h(g-g1)+h(-g-g0)-h(-g-g1)]/W
    for j, sg, t, sgn in ((neurons[0], 1, g0, -1), (neurons[1], 1, g1, +1), (neurons[2], -1, g0, -1), (neurons[3], -1, g1, +1)):
        rows.append((j, k * (sg * wg - t * one), wd, D * s * sgn / (k * W)))
    _write(sd, L, rows)
    return dict(neurons=neurons, cb=cb, g0=g0, g1=g1)

def holdcomp(sd, astd, knots=((22.0, 0.10),), center=0.0, k=0.5, G=1.0, L=15, b=1.0,
             readout=HERE / 'data/readout_L15.npz', neurons=None, rmsf=RMSF, nofs=0):
    """State-only hold compensation (odd in s - center):  dP = sum_i a_i * [h(u - s_i) - h(-u - s_i)],  u = s - center,
       h = soft relu. Undoes the model's inward hold compression beyond the trained pan range (2 neurons per knot)."""
    r = np.load(readout); ws, one = r['s'], r['one']; ws = ws - center * one
    if neurons is None: neurons = [int(j) for j in r['neurons'][nofs:nofs + 2 * len(knots)]]
    D = pan_direction(sd, astd, rmsf); s = G / float(astd[0]); rows = []
    for i, (si, ai) in enumerate(knots):    # q = -dP*s
        rows.append((neurons[2 * i], k * (ws - si * one), b * one, -D * s * ai / (k * b)))
        rows.append((neurons[2 * i + 1], k * (-ws - si * one), b * one, +D * s * ai / (k * b)))
    _write(sd, L, rows)
    return dict(neurons=neurons, knots=knots)

def sp6_factor(sd, astd_unused=None):
    """Return (q, w, beta) with sp6's readout change = q (rows 1..3, unit) x (w.h + beta)  (rank 1 because proj=URAD).
    Computed by running servo's amp_servo on a copy (deterministic, same call as agents/servo/ed/sp6)."""
    import sys; sys.path.insert(0, str(HERE.parent / 'servo')); import sv
    tmp = {'model.action_out_proj.weight': sd['model.action_out_proj.weight'].clone(), 'model.action_out_proj.bias': sd['model.action_out_proj.bias'].clone()}
    X = sv.amp_servo(tmp, 6, 0.7, 10, 30, proj=sv.URAD)             # (3, 721)
    X = X.numpy() if hasattr(X, 'numpy') else np.asarray(X)
    U_, S_, Vt = np.linalg.svd(X, full_matrices=False)
    q = U_[:, 0]; wb = S_[0] * Vt[0]
    return q, wb[:720], float(wb[720]), S_

def gated_servo(sd, astd, g0=0.065, g1=0.095, k=None, gate='xc', L=15, readout=HERE / 'data/readout_L15.npz',
                neurons=None, nofs=6, rmsf=RMSF, gain=1.0):
    """sp6's radial servo term, active only when the cube is outside the band:
         dv[rows 1..3] = gain * q * t * rho(|xc|),   t = w.h + beta  (sp6's readout), read at the L15 MLP input ('t6').
    4 hinge neurons; output direction solved by pinv (0 effect on pan, roll, gripper)."""
    r = np.load(readout); wt, one, wg = r['t6'], r['one'], r[gate]
    q = r['t6_q']
    if neurons is None: neurons = [int(j) for j in r['neurons'][nofs:nofs + 4]]
    if k is None: k = 8.0 / (g1 - g0)
    Wout = sd['model.action_out_proj.weight'].double().numpy(); gam = sd[f'{EXP}.norm.weight'].double().numpy()
    dv = np.zeros(6); dv[1:4] = q
    D = rmsf * np.linalg.pinv(Wout[:6] * gam[None, :]) @ dv; W = g1 - g0
    rows = []   # coefficient on D = gain * t * rho
    for j, sg, t, sgn in ((neurons[0], 1, g0, +1), (neurons[1], 1, g1, -1), (neurons[2], -1, g0, +1), (neurons[3], -1, g1, -1)):
        rows.append((j, k * (sg * wg - t * one), wt, D * gain * sgn / (k * W)))
    _write(sd, L, rows)
    return dict(neurons=neurons)

def sp6_apply(sd):
    """Apply servo's sp6 readout edit (same call as agents/servo/ed/sp6/EDIT.py) to sd in place."""
    import sys; sys.path.insert(0, str(HERE.parent / 'servo')); import sv
    return sv.amp_servo(sd, 6, 0.7, 10, 30, proj=sv.URAD)

def gated_hold(sd, astd, gamma=0.043, c0=0.0, g0=0.065, g1=0.095, k=None, gate='xc', G=1.0, L=15, a=1.0, b=1.0,
               readout=HERE / 'data/readout_L15.npz', neurons=None, nofs=10, rmsf=RMSF):
    """In-band export-like hold pull:  dP = -gamma * (s - c0) * (1 - rho(|g|))   (state-only; 2 exact + 4 hinge neurons).
    pan17's exact hold lacks the export's slight inward pull in band (holdgrid: export - pan17 = -0.043*s per chunk)."""
    r = np.load(readout); ws, one, wg = r['s'] - c0 * r['one'], r['one'], r[gate]
    if neurons is None: neurons = [int(j) for j in r['neurons'][nofs:nofs + 6]]
    if k is None: k = 8.0 / (g1 - g0)
    D = pan_direction(sd, astd, rmsf); sc = G * gamma / float(astd[0]); W = g1 - g0
    rows = [(neurons[0], a * ws, b * one, D * sc / (a * b)), (neurons[1], -a * ws, -b * one, D * sc / (a * b))]
    for j, sg, t, sgn in ((neurons[2], 1, g0, -1), (neurons[3], 1, g1, +1), (neurons[4], -1, g0, -1), (neurons[5], -1, g1, +1)):
        rows.append((j, k * (sg * wg - t * one), ws, D * sc * sgn / (k * W)))
    _write(sd, L, rows)
    return dict(neurons=neurons, gamma=gamma)
