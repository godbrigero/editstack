# S25 — Temporal ensembling of overlapping chunks (execution-side)

Status: untested idea
Last updated: 2026-11-10

Not a weight edit. Listed so agents don't treat it as "failed": it was added and removed on 2026-11-07 before any
measurement. The measured failures (9/24, 15/24) belong to the horizon test (S17).

## When to use
- Chunk-boundary jumps dominate the remaining jitter after S01 (hybrid: boundary jump 1.41–1.70 vs within-chunk 0.6).
- Only if the deployment code may be changed (the edit library otherwise changes weights only).

## Preconditions & confirming probes
- Measure boundary vs within-chunk second differences on stored trajectories (S11 style, no sim).
- Inference cost: one chunk ≈ 0.5 s on the laptop (~15 ticks at 30 Hz); ensembling needs a chunk per tick or per k ticks.

## Procedure
Run the policy every k steps (k < 50); keep all chunks that cover the current step; execute the exponentially weighted
average a_t = Σ_i w_i a_t^(i) / Σ w_i, w_i = exp(−m·age_i). Start with k = 10, m = 0.1.

## Verification
Paired near + ID, ≥3 seeds, against the default; watch "N" (no attempt) — S17 showed frequent re-planning can stop
the policy committing. Report boundary jitter too.

## Expected effect & cost
Unknown. Risk as in S17. Cost: k× more inference.

## Failure modes / interactions with other strategies
- Averages away the deliberate first-chunk aim and gated circuit corrections that act per chunk (S05, S15).

## Evidence log
- 2026-11-07 06:13–06:28Z, SmolVLA: added and reverted at the user's request, never measured. Evidence: scratchpad review/REVIEW.md §1.1; scratchpad lead.md §5 item 10.
