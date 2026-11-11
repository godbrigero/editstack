# S17 — Re-plan / horizon tweaks (execute fewer steps per chunk) — ANTI-STRATEGY

Status: failed
Last updated: 2026-11-10

Temporal ensembling is a different idea and was never measured: see S25 (untested).

## When to use
Don't, as a first move. Agents and users reach for it when the policy "reacts too slowly" or "drifts within a chunk".
Those symptoms had weight-level causes (S01 jitter, S04 low servo gain, S13 unaimed roll).

## Preconditions & confirming probes
If you still want to test it (e.g. the deployment rate differs from training): first measure what the policy does at
chunk boundaries (boundary jump metric) and whether failures are "N" (no attempt). Executing fewer steps changes the
input distribution the policy sees mid-chunk.

## Procedure
Config-only clone of the checkpoint with `n_action_steps` changed (10 or 25 of 50); `cp -c` clones; run the paired sets.

## Verification
Paired against the default horizon on near + ID, ≥2 seeds; look at N (no attempt) counts.

## Expected effect & cost
- cf3p25 [measured cameras], near PICK: n50 (default) 21/24; **n25 15/24; n10 9/24**. ID seed 0 PICK: n25 12/20,
  n10 8/20 (7 "N" no-attempt) vs cf3p25 19/20.
- Earlier (2026-11-07 06:07–06:28Z) "re-plan ASAP" and temporal ensembling were tried at the user's request and fully
  reverted before any measurement.
- Cost: ~1 h queue, 96 episodes; it also wrote two full checkpoint copies.

## Failure modes / interactions with other strategies
- More frequent re-planning = the policy never commits to the approach (many N).
- Any variant must be re-qualified; earlier qualifications assumed the default horizon.

## Evidence log
- 2026-11-10, SmolVLA cf3p25 [measured], horizon n10/n25. Evidence: OLD/agents/horizon/{jobs.txt,q.log,runs/*} (bench.py totals: near n10 P5+4, n25 P6+9 of 24; ID n10 P8/20, n25 P12/20); cf3p25 baseline OLD/agents/seek/REPORT.md §3.
- 2026-11-07, re-plan/TE churn reverted unmeasured. Evidence: scratchpad lead.md §5 item 10; scratchpad review/REVIEW.md §1.1.
