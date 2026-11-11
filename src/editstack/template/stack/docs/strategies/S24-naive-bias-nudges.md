# S24 — Naive output-bias nudges and time-feature shifts on an iterative sampler — ANTI-STRATEGY

Status: failed
Last updated: 2026-11-10

## When to use
Don't nudge an output bias to move a flow-matching/diffusion policy's actions. Use the exact translation (S02).

## Preconditions & confirming probes
If the head is iterative (Euler steps that feed the output back), a bias on the velocity is integrated and transformed
nonlinearly through the steps; the final shift is not the bias times a constant.

## Procedure
(What was done:)
- `out_bias`: db = −Δ/σ/0.1 on the velocity bias (assumed an ideal last step).
- `time_shift(δ)`: exact rotation of the (sin, cos) time-feature pairs so the network sees τ − δ (meant to sharpen the
  denoising contraction, 0.785 at τ = 0.1).

## Verification
Offline single-chunk probe (S09) caught both in minutes.

## Expected effect & cost
- out_bias for target 1.5/1.5/−3.1 moved joints **6.2/8.5/−14.9**; translate_actions hit the target exactly.
- time_shift on top of pad-zero overshot (pad-zero already restores the gain); dropped before sim.

## Failure modes / interactions with other strategies
- time_shift is exact algebra but the wrong mechanism; exactness does not make an edit right.

## Evidence log
- 2026-11-05, SmolVLA step-6000. Evidence: OLD/edits.py (`out_bias`, `time_shift`); lead b078391b 2026-11-05T23:19–23:32Z; scratchpad phase01.md §3.
