# S29 — Prune (zero) dead units, verified, before reusing them

Status: untested idea
Last updated: 2026-11-10

## When to use
- Before building circuits (S05) in "spare" units: zero the candidates first and prove they are dead on YOUR suites,
  instead of overwriting them blind.
- To find out how much capacity is idle in a layer (a map for later circuits).

## Preconditions & confirming probes
- Importance per unit over captured states covering all phases (REST, reach, hover, grasp, lift, closed-loop):
  importance_j = mean|silu(g_j·x)(u_j·x)| × ‖down[:, j]‖. Past: bottom 0.2–0.5 % at L15 had 0.0044–0.008 vs median
  0.025–0.047, and overwriting them left unrelated chunks unchanged within 0.1 units — suggestive, never tested as pruning.
- Exclude claimed units (`claims.claimed_indices`).

## Procedure
1. Rank units (`ops.unit_usage` weight proxy, or activation importance via `score=`).
2. For the bottom-k: `ops.zero_rows(W, gate, idx)`, `ops.zero_rows(W, up, idx)`, `W[down][:, idx] = 0` (or zero_cols).
3. Offline: max |Δchunk| over all recorded observations of the id + ood suites; keep k where it stays ≤ the
   noise-to-noise spread.
4. Record the verified-dead list in MODEL_MAP.md; later circuits claim from it.

## Verification
Offline only should suffice for k small; if k is large, run rung b–c (pruning must be behaviour-neutral).

## Expected effect & cost
Behaviour-neutral by design. Value: a verified pool of units for S05, avoiding the registry error that listed used
units (1893/1701) as free.

## Failure modes / interactions with other strategies
- Rare-phase units (lift, regrasp) look dead on reach-only captures; capture every phase.
- In a training loop, zeroed units get gradient and come back (S18).

## Evidence log
- 2026-11-07, L15 bottom-0.2 % overwrite check ≤ 0.1 units (related). Evidence: OLD/agents/circuit/REPORT.md §1.
