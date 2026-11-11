# S19 — Global output gain (scale all output rows) — ANTI-STRATEGY

Status: failed
Last updated: 2026-11-10

## When to use
Don't. If a gain is too low, find WHICH axis and WHICH region (S09 κ per axis and per region), then use S03 + S06.

## Preconditions & confirming probes
The tempting probe: an output-gain sweep reduces the "residual-on-noise" coefficient (0.027 → 0 at gain ~1.25). That
measures noise contraction, not task correctness.

## Procedure
(What was done: `action_out_proj` weight and bias × s for all 6 dims, `OLD/edits.py out_gain`.)

## Verification
n/a.

## Expected effect & cost
- g120 (×1.2): jitter 0.910 → 0.396 but **6/20 vs base 8/20** (pad-zero alone: 17/20).
- Hybrid c1_padzero_gain (pad-zero + ×1.028): ID REACH 4 → 3, loss +7.7 %/+8.4 %; not applied.
- Cost: ~4 min of sim, but it pointed at the wrong mechanism.

## Failure modes / interactions with other strategies
- Multiplies every axis, including the in-band aim and the hold, so it trades one error for another (cf. S03's
  in-band cost, which a per-axis gain already has).

## Evidence log
- 2026-11-05, SmolVLA step-6000, `g120`: 6/20, jitter 0.396. Evidence: OLD/runs/g120; lead b078391b 2026-11-05T23:10Z.
- 2026-11-07, hybrid step 500, `c1_padzero_gain`. Evidence: HYB/EDIT_LEDGER.md "c1_padzero_gain".
