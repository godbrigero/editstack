# S21 — Head/neuron interpretability hunts as the first move — ANTI-STRATEGY

Status: failed
Last updated: 2026-11-10

The comparison behind this is confounded (different model, checkpoint and test), but within each project the
interpretability-first edits did not raise scores, while interface/gain analysis did.

## When to use
Not first. Do interpretability AFTER you have read the loss masks, measured per-axis gains, and know which stage fails
(ANALYSIS_GUIDE.md order). Then it is useful to LOCATE a site (S04/S07/S16 used patching to choose layers).

## Preconditions & confirming probes
Ask first: are inputs/outputs used as trained (S01)? Is the quantity decodable (R² 0.87–0.99 everywhere → perception is
fine)? Is the read-out gain low (κ 0.03–0.6)? If yes, the fix is at the interface, not in a head.

## Procedure
n/a (see ANALYSIS_GUIDE.md probe catalogue).

## Verification
n/a.

## Expected effect & cost
- Codex line (step-2000, strict 10-case 18 s contact test): ~14 h of norm gains, attention head balance, RoPE pairs,
  Q/K directions, connector contrast, block restores; 148 runs / 1,402 rollouts; plateau at 5/10 for ~13.7 h.
- Claude line (step-6000, 20-episode held ≥ 1 s test): interface route (S01 + S02 + S08 + S09) → 17/20 in 47 min (pad0), 19/20 in 74 min (deep8, one seed).
  Different checkpoints and tests: not a controlled comparison.
- Old project: head-output patching found the servo signal distributed (each head ≤ 0.16); latent's head maps were
  correct but its edit gave no pick gain (S16).

## Failure modes / interactions with other strategies
- Distributed codes: no single head/neuron to edit; edits then become many small, unexplainable changes.

## Evidence log
- 2026-11-05, Codex sparse-resume line. Evidence: CODEX/sparse-resume-20261105/experiment_index.json; scratchpad phase01.md §1, §5.
- 2026-11-07, servo head patching. Evidence: OLD/agents/servo/REPORT.md §2.
- 2026-11-10, audit: comparison confounded. Evidence: scratchpad review/REVIEW.md §1.1.
