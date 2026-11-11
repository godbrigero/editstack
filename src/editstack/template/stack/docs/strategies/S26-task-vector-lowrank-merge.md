# S26 — Low-rank task-vector arithmetic from a related checkpoint

Status: untested idea
Last updated: 2026-11-10

## When to use
- You have a related checkpoint with the same architecture and a shared ancestor (base model, another fine-tune,
  an earlier/later training step; the hybrid project kept control checkpoints every 500 steps), and it is better at
  something the current model lacks (robustness, a camera setup, a skill).
- You want a coarse, global change that circuits cannot express, and you can afford to re-qualify everything.

## Preconditions & confirming probes
- Same tensor names/shapes (`editstack decode` INVENTORY of both). Same normalisation stats (or translate with S02/S03).
- Per-tensor task vector τ = W_other − W_ancestor; check its spectrum (`np.linalg.svd`): proceed per tensor only if a
  few singular values carry most of ‖τ‖ (e.g. ≥ 80 % in rank ≤ 16).
- Behavioural probe (S09) of both checkpoints on the same recorded observations: the target behaviour must differ.

## Procedure
1. For each candidate tensor: τ_r = `fit.truncate_rank(W_other − W_anc, r)`; save to `edits/<v>/data/`.
2. `ops.add_delta(W, name, α·τ_r)`; start with α = 0.5 on the expert/action tensors only, never on tensors that later
   edits (S05 units, S07 maps) have claimed — or add the task vector BEFORE them in the chain and refit their readouts.
3. Sweep α offline only (S09 chunk probes), not in sim.

## Verification
Offline: chunk difference vs both endpoints on recorded observations; rungs b–d; refit downstream readouts.

## Expected effect & cost
Unknown. Nearest evidence: the Codex line's "checkpoint_delta" (restoring whole blocks from the pretrained model,
200–290 k values, full rank) scored 0–4/10 — that is a block restore, not a low-rank task vector.

## Failure modes / interactions with other strategies
- Invalidates downstream readouts and gated circuits (they read activations this changes); put it first in the chain.
- Merging two fine-tunes from different data can average away both skills.

## Evidence log
- 2026-11-05, Codex line, checkpoint_delta block restores 0–4/10 (related, not this method). Evidence: scratchpad phase01.md §1; CODEX/sparse-resume-20261105/experiment_index.json.
