# S18 — Editing during training (interleaving hand edits with gradient segments) — ANTI-STRATEGY

Status: failed
Last updated: 2026-11-10

## When to use
Don't. Train to completion, then edit the finished checkpoint once (that is what produced the gain).

## Preconditions & confirming probes
If someone insists: understand the three mechanisms below and pre-register a CONTROL (same seed, no edits) and an
END-EDIT arm (control + the same recipes applied once at the end).

## Procedure
(What was done: pause every 500 steps, eval, apply closed-form edits, keep if they pass rules, resume; Adam freeze
for circuit units: exp_avg = 0, exp_avg_sq = max(v, 1e30) via `HYB/tools/adam_freeze.py`.)

## Verification
Paired three-arm comparison at the final step (HYB/agents/arm3/ARM3.md).

## Expected effect & cost
- Step 6000, main suite /80 (paired, 2 seeds): HYBRID 39/80 REACH, CONTROL 37/80, **ENDEDIT 60/80**.
  ENDEDIT − HYBRID +26.2 ± 6.8 pp REACH, +16.2 ± 7.2 PICK. HYBRID − CONTROL ≈ 0 (moving cube −25 ± 10.9, worse).
- ENDEDIT's own costs: held-out loss +27 %, boundary jumps +19 %, 0/16 under a pole occluder.
- Cost: 45.7 h orchestrator span, ~34 h of interventions, ~35 agents; 10 of 17 new-edit attempts rejected/null.

## Failure modes / interactions with other strategies
1. **Training cancels an edit's output on training frames**: c2_tipservo_v2's weights stayed 99 % intact but its
   behaviour washed out (REACH 27 → 7); the gradient moved upstream features to "undo the edit" (cos 0.84).
2. **Small weights drift under Adam**: dead-zone circuit up/down weights (~1e-3, bf16) took full-size steps (10–17× √v)
   from the 3–5 % of tokens outside the dead zone → circuit dead by the next checkpoint. Freezing fixed weights only.
3. **Frozen circuits, drifting inputs**: the L15 residual drifts (cube probe loses 38 % slope per segment); circuits
   switch off or become load-bearing (removal +702 % loss at 5000) and misfire.
- The first two, simplest edits (S01 + S04) delivered all the main-suite gain; the 4 later circuits added 0 (S12).

## Evidence log
- 2026-11-07 → 11-09, hybrid SmolVLA (smolvla_base fine-tune, 50 demos), 12 cycles. Evidence: HYB/REPORT.md §1–3; HYB/EDIT_LEDGER.md (c2, c3, c4, ENDEDIT); HYB/agents/arm3/ARM3.md.
