# S02 — Exact translation of the output space (flow-matching aware)

Status: partial
Last updated: 2026-11-10

The algebra is proven exact (time-feature fit error 1.7e-8, target shift hit exactly). A closed-loop gain beyond
seed noise was never demonstrated (deep8: 17/20 → 19/20 on one seed). Use it when a measured offset needs removing;
do not expect it to be the win by itself.

## When to use
- Signed error vs the task variable is FLAT with a constant offset (e.g. gripper always closes 6–10 mm short along
  the jaw axis; real arm consistently overshoots by a few joint units).
- A calibration offset between sim and hardware (base yaw offset, joint zero offsets).
- As the bias part of S03 (scale about a centre other than the mean).

## Preconditions & confirming probes
1. Measure the offset in task space from stored trajectories (`OLD/geo.py`: tip-to-cube offset at close).
2. Convert it to joint/action units with the Jacobian (S08). Past: 8 mm deeper along the gripper axis =
   Δ_raw = [0, +1.52, +1.52, −3.11, 0, 0] (shoulder_lift, elbow, wrist_flex).
3. Know the sampler. SmolVLA flow matching: x_τ = τ·ε + (1−τ)·y, network predicts v ≈ ε − y, 10 Euler steps
   τ = 1.0 … 0.1. Read the time embedding code: φ(τ) = [sin(ω τ), cos(ω τ)], ω_k = 2π / (p_min (p_max/p_min)^{f_k}),
   f = linspace(0,1,360), p_min = 0.004, p_max = 4.0 (720 features).
4. A naive output-bias nudge does NOT work for an iterative sampler (S24): it moved joints 6.2/8.5/−14.9 against a
   1.5/1.5/−3.1 target.

## Procedure
Goal: every sample becomes y + Δ, and the network keeps seeing its usual internal frame.
Normalised shift d = Δ_raw / σ_action (32-d, zeros on padding); state shift d_s = Δ_raw / σ_state.

Algebra. With y' = y + d: x'_τ = x_τ + (1−τ) d. The network must see x'_τ − (1−τ) d = x_τ and output v' = v − d.
- Action input projection A (y = A x + b_in): A(x' − (1−τ)d) + b_in = A x' + (b_in − A d) + τ·(A d).
  - Constant part: `b_in -= A d`.
  - τ part: the action embedding is concatenated with φ(τ) into `action_time_mlp_in` (W1 = [W_a | W_t], 720×1440).
    Adding τ·(A d) to the action half adds τ·W_a A d to the pre-activation. Fit c (720) with φ(τ)·c ≈ τ
    (ridge 1e-6 on a 2001-point grid of [0,1] plus the 10 inference τ repeated 200×; error at inference τ ≤ 1.7e-8),
    then `W_t += outer(W_a A d, c)`.
- Output: `b_out[:6] -= d[:6]` (v' = v − d).
- State (optional, recommended): the state input moves by d_s too, so `b_state -= W_state[:, :6] d_s[:6]`.

editstack ops (W is float64; `A`, `W_a` read BEFORE editing):
```python
AIN, AINB = "model.action_in_proj.weight", "model.action_in_proj.bias"
TM = "model.action_time_mlp_in.weight"            # cols 0..719 action emb, 720..1439 time features
AOB = "model.action_out_proj.bias"; SPW, SPB = "model.state_proj.weight", "model.state_proj.bias"
Ad = W[AIN] @ d                                    # (720,)
ops.compensate_input_shift(W, AIN, AINB, slice(None), d)            # b_in -= A d
c = ctx.data("tau_coeffs.npy")                     # from a probe script, see OLD/edits.py tau_coeffs()
ops.rank1(W, TM, u=W[TM][:, :720] @ Ad, v=np.r_[np.zeros(720), c])   # W_t += outer(W_a A d, c)
ops.shift_bias(W, AOB, slice(0, 6), -d[:6])
ops.compensate_input_shift(W, SPW, SPB, slice(0, 6), d_s[:6])       # state seen in its usual frame
```
Reference implementation: `OLD/edits.py translate_actions` (+ `tau_coeffs`). Report the τ-fit error in the docstring.

## Verification
- Rung a: offline on recorded observations with fixed noise: the edited chunk equals original chunk + Δ_raw on every
  step to float tolerance (it is an exact identity, not an approximation), and the τ-fit error is ≤ 1e-6.
- Then measure the geometric offset closed loop (the quantity you meant to fix), not only success.
- Rungs b–d as usual; a few-mm change will not separate from seed noise on 20 episodes — use ≥3 seeds and the
  geometric error as the primary metric.

## Expected effect & cost
- deep8 (8 mm deeper): 17/20 → 19/20 on one seed (within noise; only 3/20 still held at the episode end in both).
- rad4 (+4 mm radial on top): 17/20 (worse, one seed).
- Real arm: `rt1` (lift −0.8, elbow +5.0, wrist flex +5.8 to fix real overshoot) 2/2 at centre on hardware, but
  ID REACH 15/40, PICK 11/40 in the corrected (measured-camera) sim — it was tuned on stale sim renders.
  `t5` (pan −5 units, real base offset) on cf1/cf3p25 picked on the real arm (1/1 at ±10 cm each; tiny n).
- Cost: minutes; 4 tensors touched (`action_in_proj` bias, `action_time_mlp_in` time half, `action_out_proj` bias,
  `state_proj` bias).

## Failure modes / interactions with other strategies
- Translating the state too is a choice: with it, the hold point moves with the commands (what you usually want).
- Stacks with S03 (same algebra; scale_axis uses this τ term for its centre offset) and everything downstream.
- A translation tuned on a mis-specified harness transfers the harness error (rt1). Audit the harness first.

## Evidence log
- 2026-11-05, SmolVLA step-6000, naive `out_bias`: joints moved 6.2/8.5/−14.9 vs target 1.5/1.5/−3.1. Evidence: lead b078391b 23:29Z; scratchpad phase01.md §3.
- 2026-11-05, `pad0_deep8`: 19/20 vs pad0 17/20 (1 seed), τ-fit error 1.7e-8. Evidence: OLD/ed/pad0_deep8/EDIT.py, OLD/runs/pad0_deep8.
- 2026-11-05, `pad0_deep8_rad4`: 17/20. Evidence: OLD/ed/pad0_deep8_rad4/EDIT.py.
- 2026-11-09, `rt1` real-arm translation: 2/2 real; measured-cam sim ID R15/P11 of 40. Evidence: lead b078391b 2026-11-09T16:33Z; OLD/candidates.json "rt1" (bench.py totals).
- 2026-11-09, `cf1t5`, `cf3p25t5` (pan −5): real-arm picks (1/1 per placement). Evidence: scratchpad lead.md §5 (secondary).
- 2026-11-10, audit: deep8's gain is single-seed noise; method confirmed. Evidence: scratchpad review/REVIEW.md §1.1.
