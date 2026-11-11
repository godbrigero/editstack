# S11 — Rescore stored trajectories when the metric changes (and use a staged metric)

Status: proven
Last updated: 2026-11-10

## When to use
- The success definition changes (it changed twice in the old project), or you suspect a binary metric hides the
  stage an edit fixed.
- Before declaring an edit "null": was its stage visible in the metric?

## Preconditions & confirming probes
- Every eval stores per-episode trajectories (`traj_<ep>.npz`: actions, states, geo = cube/tip/jaw xyz) and a
  `results.jsonl` line (max_lift, held_1s, yaw, ...). Without these, rescoring means re-running the sim.

## Procedure
1. Write the new metric as a pure function of stored arrays (`OLD/rescore.py`: pick = max_lift ≥ 2 cm;
   `OLD/bench.py`: staged labels).
2. Staged metric (bench.py): find the first close after opening; within 0.5 s take the closest tip-to-cube approach.
   P = lifted ≥ 2 cm; R = tip within 1.5 cm horizontally and 1 cm vertically but no pick; s/l/w/h = short/long/wide/high;
   N = no attempt. REACH = P + R. Calibrate thresholds on stored successes (99.7 % of 825 picks closed within 1.5 cm).
3. Rescore ALL past runs of all candidates, re-rank, and re-read earlier "null" reports under the new metric.
4. Record the metric version with every leaderboard entry.

## Verification
- Spot-check labels on a handful of episodes against geometry (and video if needed).

## Expected effect & cost
- Seconds per run dir. Under lift ≥ 2 cm the export's ID went to 73/80 and pan17's ID cost became visible (~2.2 SE).
- Under REACH, latent's r1a went from "9/12 picks, keep the baseline" to 23/24 REACH; r1a_xg became best-thus-far.
  Wave-1 "nulls" (latent, circuit) were largely a metric that could not see the stage they fixed.
- The staged metric exposed that remaining failures were grasp (R), redirecting work to S10/S13.

## Failure modes / interactions with other strategies
- Changing the metric mid-run invalidates comparisons with unrescored numbers; keep both columns.
- A lenient stage metric can hide a real-world failure (REACH counts "reached but squeezed out").

## Evidence log
- 2026-11-07 02:51Z, user: lifted ≥ 2 cm counts → rescore.py. Evidence: OLD/rescore.py; scratchpad lead.md §2.
- 2026-11-07 03:44Z, REACH counted as success → bench.py (calibrated on 825 episodes). Evidence: OLD/bench.py docstring.
- 2026-11-10, audit: wave-1 nulls were a metric artefact. Evidence: scratchpad review/REVIEW.md §1.1.
