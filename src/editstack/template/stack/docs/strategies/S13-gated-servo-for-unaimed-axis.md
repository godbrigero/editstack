# S13 — Gated servo for an axis the model never aims (wrist-roll alignment)

Status: partial
Last updated: 2026-11-10

Mechanism verified (misalignment at close 20.3° → 15.9°); the pick gain is within noise (lead: "alignment improves,
picks don't"). Keep it only as a cheap, exact link; do not count on it.

## When to use
- Failure taxonomy points at one secondary axis: 49 of 69 "reached but not picked" episodes had jaw/cube
  misalignment ≥ 25° (mod 90°); P rate 0–10° 98 %, 35–46° 59 %.
- κ on that axis ≈ 0 (roll κ ≈ 0.03; roll at close = 0.15·roll_IK + noise) — nothing to amplify, so S03 is out.
- The needed quantity IS linearly encoded inside (relative jaw/cube angle at L15 MLP input: paired sign 92–94 %,
  zero-point error ±16.5°).

## Preconditions & confirming probes
1. S10 harness upper bound: aligned start 40/42 vs natural 33/42 (sp6).
2. Readouts at L15 MLP input: angle a (deg, nearest jaw alignment), gripper state g (rmse 2.1 units), tip height h
   (0.5 cm), `one`.
3. Noise-coupling projection (S05 precondition 4): without it the roll readout kicked 50–2000 units.

## Procedure (`OLD/agents/grasp/grs.py gated_roll2`)
1. Gate variable z = (g − 85) + 8.5·(h − 4.5): on (z > 8) when jaws open AND tip ≥ ~4.5 cm; off (z < 0) in the final
   descent, closing, lifting and at REST (z ≈ −12).
2. Two hinges: unit 1 gate = k·z, up = a, down = D·q/(8k); unit 2 gate = k(z − 8), up = a, down = −D·q/(8k);
   k = 1, q = −gain; D = roll-only direction (S05 step 1, joint 4).
   ⇒ dv_roll = −gain·a·ramp(z; 0 → 8). gain = 0.020 (grh20).
3. Use unclaimed units (grasp used 779/1893; on p17gsh approach's units were moved to 1263/248; grh20_gs used 1893/1701).

## Verification
- Offline gate values: open hover at 4.5/7/10 cm → 0.79/1/1; ≤ 2.5 cm, closed, lifted, REST → 0.
- Harness (S10) then closed loop; mechanism metric = misalignment at close, not just PICK.

## Expected effect & cost
- Harness: grh20 36/42 vs 33/42. Closed loop on sp6: near PICK 31/36 (30), ID PICK 31/40 (28).
- On p17gsh → p17gshr: near PICK 29 → 32/36, ID unchanged; on pan17: −1 near / −4 ID; p17gs: 0 / −1.
  Net over 4 bases: near +3, ID −2 → within noise.
- Cost: 2 units, one readout capture.

## Failure modes / interactions with other strategies
- Ungated or gripper-only-gated roll servos rotate the jaws while straddling the cube and push it (grs25 29/42).
- The linear angle readout's ±11–16° pose bias and only 1–2 chunks of hover above 4.5 cm leave many episodes 25–40° off.
- The tip is 1.2 cm off the roll axis: rolling shifts the tip (new R cases from 1–1.3 cm shifts; approach's ep7).
- Unit collision with S05/S06 circuits (779/1893).
- Next idea (untested): correct the roll in the first chunk (gate "far from cube") so no motion near the cube is needed.

## Evidence log
- 2026-11-07, SmolVLA sp6 base, `grh20`: harness 36/42; near P31/36, ID P31/40. Evidence: OLD/agents/grasp/REPORT.md §1, §4; OLD/agents/grasp/ed/grh20.
- 2026-11-07, `p17gshr` (approach stack): near P29 → 32/36. Evidence: OLD/agents/approach/REPORT.md.
- 2026-11-07, lead: "Alignment improves, picks don't" (10:53Z). Evidence: lead b078391b; scratchpad review/REVIEW.md §1.1.
