# S01 — Disconnect never-supervised inputs (pad-zero)

Status: proven
Last updated: 2026-11-10

Path prefixes used in this library: `OLD` = /Users/godbrigero/Documents/smolvla-weight-edit-6000,
`HYB` = /Users/godbrigero/Documents/hybrid-train-smolvla, `CODEX` = /Users/godbrigero/Documents/yellow-cube-manual-weights-20261104.

## When to use
- High-frequency wiggle inside each output chunk (second difference of commanded actions within a chunk is large),
  while chunk-to-chunk behaviour is roughly right.
- The model pads an input/output to a fixed width (e.g. 6 joints padded to 32), or has slots that are fed at
  inference but carry no supervised signal.
- Generative / iterative heads (flow matching, diffusion, autoregressive feedback) are the most exposed, because the
  unsupervised channels are fed back at every step.

## Preconditions & confirming probes
1. **Read the loss and its mask.** SmolVLA: `losses[:, :, :original_action_dim]` (only dims 0–5 supervised). If the
   padded outputs are not in the loss, their values at inference are untrained noise.
2. **Check the input weights of the padded channels are not ~0.** SmolVLA `model.action_in_proj.weight[:, 6:]` had
   column norms ~2.7–3.3, the same as the joint columns (inherited from the base model).
3. **Measure the distribution gap.** At the last Euler step the padding input std was ~0.59 vs 0.1 in training (~6× OOD).
4. **Single-chunk probe** (S09): on recorded observations with fixed noises, compare within-chunk second difference
   (`d2`) for original vs columns zeroed. Past: d2 0.465–1.976 → 0.085–0.298 per joint (~5×).
5. Do NOT substitute "zero the initial padding noise": it is not equivalent (outputs differed by up to 0.22),
   because the network itself writes non-zero values into those channels at every step. Only the weight cut removes the path.

## Procedure
1. Identify the consumer of the padded channels (here `model.action_in_proj.weight`, shape [720, 32], y = x Wᵀ + b).
2. In `edits/<v>/edit.py`:
   ```python
   ops.zero_cols(W, "model.action_in_proj.weight", slice(6, None))   # 720 x 26 = 18,720 values
   ```
   That is the whole edit. Mathematically it turns the head into a true 6-dim head.
3. If the padded channels ALSO feed something else (state padding into `state_proj`, etc.), check each consumer
   separately with the same loss-mask argument before zeroing.
4. Docstring: hypothesis (loss mask line), measured d2 before/after, expected effect.

## Verification
- Rung a: `editstack apply <v>` shows one tensor, 18,720 values changed. Offline probe: d2 drops by the predicted
  factor on recorded observations; the chunk MEAN on supervised dims barely moves (it should not change behaviour,
  only noise).
- Rungs b–d: screen, then qualify on ≥3 seeds. Jitter is a paired, low-noise metric: report it with the success counts.

## Expected effect & cost
- Old project (step-6000 checkpoint, 20 ID episodes, held ≥1 s): **8/20 → 17/20, mean jitter 0.910 → 0.192**.
  One noise seed for the success count, but the jitter drop is large and paired.
- Hybrid project (step 500, 80 episodes, 2 seeds paired): within-chunk smoothness 0.855 → 0.593 (−0.263 ± 0.004 SE);
  REACH 5 → 6 (n.s.); held-out loss +4.1%. On the finished control checkpoint (ENDEDIT m1): main 44/80 vs control 37/80.
- Codex line (step-2000, strict 18 s test): action-step RMS −43.8%, still 0/5 on that test.
- Cost: minutes. No data, no fit.

## Failure modes / interactions with other strategies
- Combining with a time-feature rotation (`time_shift`, S24) overshot: pad-zero already restores the denoising contraction.
- Combining with a global output gain (×1.028, S19) cost ID REACH in the hybrid.
- All later edits in both projects were built on top of it; it never conflicted.
- Gripper boundary smoothness got slightly worse in the hybrid (2.68 → 2.96); check every channel, not the mean.

## Evidence log
- 2026-11-05, SmolVLA step-6000 (OLD), `pad0`: 17/20 vs base 8/20, jitter 0.192 vs 0.910; probe p5 d2 ~5× lower.
  Evidence: OLD/edits.py `pad_zero`; OLD/runs/pad0, OLD/runs/base_ns0; lead transcript b078391b 2026-11-05T23:19–23:27Z.
- 2026-11-05, Codex step-2000 (CODEX), pad-zero: −43.8% action-step RMS, 0/5. Evidence: scratchpad phase01.md §1 (secondary).
- 2026-11-07, hybrid step 500, `c1_padzero`: smoothness −31%, REACH +1 (n.s.), KEPT. Evidence: HYB/EDIT_LEDGER.md "c1_padzero".
- 2026-11-09, hybrid control/6000, ENDEDIT m1: main 44/80 (control 37/80). Evidence: HYB/EDIT_LEDGER.md "ENDEDIT", HYB/agents/endedit/chain.log.
