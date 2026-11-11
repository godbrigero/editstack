# S03 — Per-axis affine scale with perception compensated (`scale_axis`)

Status: proven
Last updated: 2026-11-10

## When to use
- The error on one output axis GROWS with distance from the trained range (a slope, not an offset): e.g. lateral
  cubes beyond the trained band closed short in pan.
- `fit.gain` on that axis is well below 1 out of band while the task variable is still linearly decodable (R² high):
  the read-out, not perception, is under-gained.
- You want every image-driven move on that axis k× larger while holding still stays unchanged.

## Preconditions & confirming probes
1. Per-chunk correction gain κ on the axis (S09; `OLD/agents/servo/kappa.py`): past pan κ 0.48–0.68 (export),
   needed ~1. First chunk from rest: pan move / needed 0.74 out of band (0.90–0.98 in band) (`OLD/agents/pancost/aim1.py`).
2. Hold behaviour: mid-episode, commands ≈ state on that axis via the state input (`state_proj`). Measure
   d(output)/d(state) on the axis (past: 0.92–0.99). This is why the state input must be compensated: scaling only the
   output gives k·J ≈ 1.6 → runaway; scaling only the state gives J/k ≈ 0.56 → collapse to the centre (pancost §4).
3. Choose the centre C (raw units). Default = action mean. C is the fixed point of the hold.
4. Write down the in-band cost you expect: a uniform k also multiplies the in-band aim (see failure modes).

## Procedure
Wanted: commanded y' = C + k(y − C) on axis `dim`, perceived state s_eff = C + (s − C)/k.
Normalised (μ, σ action; μ_s, σ_s state): â' = k·â + b with **b = (1 − k)(C − μ)/σ**.
Flow-matching consequences (sampler as in S02): the network must see x = (x' − (1−τ) b)/k and output v' = k v − b.
1. Action input column: `A[:, dim] /= k` → call this col.
2. If b ≠ 0: translate by Ad = col·b exactly as S02 (b_in −= Ad; W_t += outer(W_a Ad, c)).
3. Output row: `W_out[dim] *= k`; `b_out[dim] = k·b_out[dim] − b`.
4. State column (original column w): `b_state += w·(C − μ_s)(1 − 1/k)/σ_s`, then `W_state[:, dim] /= k`.
Exact except one thing: the network sees noise std 1/k on that channel at τ = 1 (measured effect ≤ 0.06 units on chunk means).

editstack ops:
```python
w = W[SPW][:, dim].copy()                                   # BEFORE scaling
ops.compensate_input_scale(W, AIN, [dim], k)                # A[:, dim] /= k
if b != 0:                                                  # centre != action mean
    Ad = W[AIN][:, dim] * b
    W[AINB] -= Ad
    ops.rank1(W, TM, u=W[TM][:, :720] @ Ad, v=np.r_[np.zeros(720), c])
ops.scale_out(W, AOUT_W, k, rows=[dim], bias=AOUT_B)        # row and bias * k
ops.shift_bias(W, AOUT_B, dim, -b)
W[SPB] += w * (C - mu_s) * (1 - 1 / k) / sd_s
ops.compensate_input_scale(W, SPW, [dim], k)
```
Reference: `OLD/edits.py scale_axis`. First run `ops.rescale_channel` (the function-preserving pair) to confirm you
have the right tensors, then apply the behaviour-changing version above.

## Verification
- Rung a (exactness): `OLD/check_scale.py` pattern. Run the ORIGINAL model on (state with s_eff on `dim`, noise on
  `dim` divided by k), transform its output y → C + k(y − C), and compare with the edited model on the real input.
  Past: max diff 0.05 raw units (float rounding). Then measure κ on the axis: it must be exactly k× (servo: 0.57 → 0.97).
- Rungs b–d: both the out-of-band set AND the in-distribution set, ≥3 seeds; the in-band cost shows only on ID.

## Expected effect & cost
- pan × 1.7: near-lateral REACH 5/12 → 31/36, PICK 1/12 → 29/36 (3 seeds). ID PICK 73/80 → 63/80 (4 seeds, ≈2.2 SE).
- The ID cost is real: in-band chunk-1 aim becomes 1.46–1.56× the needed move (export 0.90–0.98); tangential error at
  close median 0.40 vs 0.23 cm. Remove it with S06 (gate the gain to out-of-band), not by tuning k.
- Cost: minutes of compute, one exactness probe (seconds), then the eval ladder.

## Failure modes / interactions with other strategies
- **No single k is right everywhere**: k 1.5/1.6 keep the ID cost (chunk-1 ID gain 1.46–1.49); k 1.3 loses near (7/12).
  A different centre only adds a constant offset (1−k)(C'−C). Do not sweep k: diagnose and gate (S06).
- Scaling the gripper axis (k 1.05 about 5 units, the clip limit) was a wash: ID 64/80 vs 63/80, near 27/36 vs 29/36.
- A roll `scale_axis` is pointless when the axis has no image servo to amplify (κ_roll ≈ 0.03, S13).
- Everything later (S05, S06, S07, S13) was built on top; readouts must be refit on the scaled base
  (`readout_L15_p17.npz` vs `_exp.npz`).
- Interacts with shadow dependence (S14): pan17 loses 19/24 → 14/24 PICK near with shadows off.

## Evidence log
- 2026-11-06, SmolVLA step-6000, `scale_axis` written and verified offline, max diff 0.05. Evidence: OLD/check_scale.py; lead b078391b 2026-11-06T23:38Z.
- 2026-11-07, `pan17` (deep8_pan17): near R31/36 P29/36, ID R76/80 P63/80. Evidence: OLD/ed/deep8_pan17/EDIT.py; OLD/candidates.json "pan17"; OLD/export/BEST_THUS_FAR.md.
- 2026-11-07, pancost k sweep (1.3/1.5/1.6), gripper ×1.05, PE warp: none removes the ID cost. Evidence: OLD/agents/pancost/REPORT.md §Results, §4, §6.
- 2026-11-07, servo κ_pan 0.57 → 0.97 under pan×1.7 (exactly 1.7×). Evidence: OLD/agents/servo/REPORT.md §1.
