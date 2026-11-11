# S27 — Norm-scale recalibration from activation statistics (diagonal S07)

Status: untested idea
Last updated: 2026-11-10

## When to use
- An input shift mostly rescales channels (brightness, exposure, resolution, normalisation stats) rather than mixing
  them: per-channel ratio of activation rms new/old varies a lot across channels but the correlation structure is kept.
- You want a cheaper, lower-capacity alternative to the full per-layer map of S07 (960² values per layer), with less
  overfitting risk.

## Preconditions & confirming probes
- Paired captures as in S07 (teacher-forced). Compute, at the input of each RMSNorm/LayerNorm consumer, per-channel
  s_c = rms_old,c / rms_new,c of the NORMALISED output. Fit the diagonal map vs the full map on held-out placements:
  if the diagonal leaves ≤ ~1.3× the residual of the full map, use it.

## Procedure
1. For each norm with weight γ (and bias β for LayerNorm), on paired samples: diag ridge per channel
   m_c = argmin Σ(y_old,c − m_c y_new,c)², shrunk to 1: m_c = (Σ y_new y_old + λ)/(Σ y_new² + λ).
   (LayerNorm: also fit an offset o_c and fold into β.)
2. Fold: γ_c ← γ_c·m_c (`ops.scale_out`-style on a 1-D tensor: `W[gamma] *= m`), β_c ← m_c β_c + o_c.
3. Equivalent form: `ops.fold_input_map(W, consumer, np.diag(m))` on the next linear layer.

## Verification
As S07: held-out fraction of shift left per layer, then rungs b–d under the new condition.

## Expected effect & cost
Unknown. Nearest evidence: the Codex line HAND-TUNED norm gains (final norm ×1.15, vision post-LN γ/β ×1.25, others
×0.9/×1.25) as part of a 5/10 recipe that plateaued — hand-tuning, not statistics matching. S07's full maps recovered
6/40 → 57–59/60, an upper bound for what a diagonal can do.

## Failure modes / interactions with other strategies
- Cannot undo mixing/rotations of features (camera geometry changes are not diagonal); prefer S07 there.
- Changing a norm upstream of S05 readouts changes their inputs: place it first, as with S07.

## Evidence log
- 2026-11-05, Codex hand-tuned norm gains (related). Evidence: scratchpad phase01.md §1 ("The 5/10 recipe").
