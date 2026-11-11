# S20 — Hard-coding a controller by rewriting hundreds of millions of values — ANTI-STRATEGY

Status: failed
Last updated: 2026-11-10

## When to use
Never. It replaces the model instead of editing its behaviour, and it does not generalise past the cases it was built
for. The user rejected it ("you just encoded a controller").

## Preconditions & confirming probes
Red flags that you are drifting into it: zeroing vision/language paths, routing on case identity or fixed placements,
changed-value counts in the 10⁷–10⁸ range, success only on the fixture you tuned on.

## Procedure
(What was done: `handwired_smooth_start`: 214,284,189 scalars changed, vision/language zeroed, fixed-placement
trajectory.)

## Verification
Counter-cases (new placements) immediately: 2/5 on the 5 fixed cases → 2/10 with 5 counter-cases.

## Expected effect & cost
- 2/5 → 2/10; both lifts dropped at 19.5 s. ~1.4 h lost, plus a restart from the original weights.
- The sparse restart that followed (< 100,000 changed values) reached 5/10 (88,332 values) and plateaued.

## Failure modes / interactions with other strategies
- Breaks every downstream analysis (the model's own perception is gone).
- Budget rules (changed-value caps) help but are not the point; use exact, explainable edits (S01–S07).

## Evidence log
- 2026-11-05, Codex line, SmolVLA step-2000. Evidence: CODEX/checkpoints/handwired_smooth_start/manual_edit_manifest.json ("changed_scalars": 214284189); CODEX/experiment_index.json (handwired_smooth_start_verified 2/5); scratchpad phase01.md §1.
