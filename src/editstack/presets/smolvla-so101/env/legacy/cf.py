from pathlib import Path
"""agent 'camfix': camera-transfer edits for the MEASURED cameras (2026-11-09).
apply_maps(sd, maps, layers, alpha): expert attention output maps A_L (o_train ~= o_meas @ A_L), folded exactly into
    lm_expert.layers.L.self_attn.o_proj.weight <- W @ (I + alpha (A_L - I))^T.   alpha < 1 shrinks the map toward I."""
import numpy as np, torch
EXP = 'model.vlm_with_expert.lm_expert'
def apply_maps(sd, maps, layers=None, alpha=1.0):
    z = np.load(maps); layers = layers if layers is not None else [int(l) for l in z['layers']]
    for L in layers:
        k = f'{EXP}.layers.{L}.self_attn.o_proj.weight'; W = sd[k]
        A = torch.tensor(z[f'A{L}'], dtype=torch.float64); A = torch.eye(A.shape[0], dtype=torch.float64) + alpha * (A - torch.eye(A.shape[0], dtype=torch.float64))
        sd[k] = (W.double() @ A.T).to(W.dtype)
    return layers

def low_pan_gain(sd, astd, beta=0.15, h_off=6.0, h_on=3.0, neurons=(301, 1703), L=15, rmsf=3.3,
                 ro_d=str(Path(__file__).resolve().parent / 'data/readout_L15_p17.npz'),
                 ro_h=str(Path(__file__).resolve().parent / 'data/grs_readout_v2.npz'), ksharp=8.0):
    """Height-gated pan-servo gain (2 soft-hinge SwiGLU neurons at expert L15):
         dP[pan units] = +beta * d * ramp(h),  ramp = 1 for tip height h <= h_on cm, 0 for h >= h_off (linear between),
       d = planned - state pan (readout 'd'), h = tip-height readout (cm, grasp agent). Exactly proportional to d, so 0 when
       holding; off at REST (h reads ~8.5) and when lifted. Restores the [measured]+maps lateral servo gain near the cube
       (kappa_tan at h = 3 cm: 0.64-0.69 vs 0.80 [training])."""
    import gpg
    rd, rh = np.load(ro_d), np.load(ro_h); wd, one, wh = rd['d'], rd['one'], rh['h']
    z1 = h_off - h_on; k = 8.0 / z1
    wz = h_off * one - wh                       # z = h_off - h  (>= z1 -> fully on)
    D = gpg.pan_direction(sd, astd, rmsf); q = -beta / float(astd[0])   # velocity coefficient per unit (d * ramp)
    rows = [(neurons[0], k * wz, wd, D * q / (k * z1)), (neurons[1], k * (wz - z1 * one), wd, -D * q / (k * z1))]
    gpg._write(sd, L, rows)
    return dict(neurons=neurons, beta=beta)

def regrasp(sd, astd, e0=0.40, e1=0.55, delta=84.0, neurons=(595, 233), L=15, rmsf=3.3,
            readout=str(Path(__file__).resolve().parent / 'data/grip_gate_cf2p25.npz')):
    """Re-open after an EMPTY close: 2 soft-hinge neurons at expert L15.
         gripper command += delta * ramp(e),  e = 'empty closed gripper, lifted' readout (1 vs 0; held-out: empty 0.97+-0.11,
         holding 0.04+-0.09 max 0.31), ramp 0 below e0, 1 above e1. Fully opens the gripper for that chunk; the policy then
         re-approaches from the hover with an open gripper. Never fires while a cube is held (e <= 0.31 < e0)."""
    import grs, gpg
    r = np.load(readout); we, one = r['e'].astype(np.float64), r['one'].astype(np.float64)
    W = e1 - e0; k = 8.0 / W
    D = grs.joint_direction(sd, 5); q = -delta / float(astd[5])     # +1 normalized velocity on the gripper -> chunk moves by -1
    rows = [(neurons[0], k * (we - e0 * one), one, D * q / (k * W)), (neurons[1], k * (we - e1 * one), one, -D * q / (k * W))]
    gpg._write(sd, L, rows)
    return dict(neurons=neurons)

def regrasp2(sd, astd, A=20.0, ec=0.55, hc=3.0, z0=3.5, z1=8.5, delta=84.0, neurons=(595, 233), L=15, rmsf=3.3,
             readout=str(Path(__file__).resolve().parent / 'data/grip_gate_real_cf3.npz'),
             ro_h=str(Path(__file__).resolve().parent / 'data/grs_readout_v2.npz'), ksharp=8.0):
    """Re-open after an EMPTY close, gated on z = A*(e - ec) + (h - hc):  e = 'empty closed gripper, lifted' readout fitted
       on REAL closed-loop states (per-token rows), h = tip-height readout (cm). gripper command += delta * ramp(z; z0 -> z1).
       On the fitting rows: holding max z = +2.5, other negatives max +1.2, empty-lifted p5 +8.9 / median +13.4."""
    import grs, gpg
    r = np.load(readout); we, one = r['e'].astype(np.float64), r['one'].astype(np.float64); wh = np.load(ro_h)['h'].astype(np.float64)
    wz = A * we + wh - (A * ec + hc) * one
    W = z1 - z0; k = ksharp / W
    D = grs.joint_direction(sd, 5); q = -delta / float(astd[5])
    rows = [(neurons[0], k * (wz - z0 * one), one, D * q / (k * W)), (neurons[1], k * (wz - z1 * one), one, -D * q / (k * W))]
    gpg._write(sd, L, rows)
    return dict(neurons=neurons)

def regrasp3(sd, astd, A=30.0, ec=0.6, bh=0.5, hc=3.0, z0=4.0, z1=9.0, delta=420.0, ksharp=32.0, neurons=(595, 233), L=15,
             readout=str(Path(__file__).resolve().parent / 'data/grip_gate_real2_cf3.npz'),
             ro_h=str(Path(__file__).resolve().parent / 'data/grs_readout_v2.npz')):
    """regrasp2 with the gate refit including REST / early-reach states as negatives (grip_gate_real2_cf3) and
       z = A*(e - ec) + bh*(h - hc); gripper command += delta * ramp(z; z0 -> z1)."""
    import grs, gpg
    r = np.load(readout); we, one = r['e'].astype(np.float64), r['one'].astype(np.float64); wh = np.load(ro_h)['h'].astype(np.float64)
    wz = A * we + bh * wh - (A * ec + bh * hc) * one
    W = z1 - z0; k = ksharp / W
    D = grs.joint_direction(sd, 5); q = -delta / float(astd[5])
    rows = [(neurons[0], k * (wz - z0 * one), one, D * q / (k * W)), (neurons[1], k * (wz - z1 * one), one, -D * q / (k * W))]
    gpg._write(sd, L, rows)
    return dict(neurons=neurons)
