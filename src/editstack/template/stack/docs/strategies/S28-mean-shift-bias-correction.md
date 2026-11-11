# S28 — Mean-shift bias correction for a distribution shift (offset-only S07)

Status: untested idea
Last updated: 2026-11-10

## When to use
- After an input shift (sensor, lighting, preprocessing), as the FIRST, cheapest thing to try before a full map (S07).
- When the shift is mostly a constant offset of some layer's activations (check below).

## Preconditions & confirming probes
- Paired captures (S07 teacher-forced). Per layer, the fraction of the shift explained by its mean:
  f = ‖E[x_old − x_new]‖² / E‖x_old − x_new‖². If f ≳ 0.5 at some layer, an offset fixes most of it there.
- The consumer must have a bias to absorb the constant. SmolVLA's expert `o_proj` has none; use the next consumer that
  has one, or a constant unit (S05 `one` readout through an exact linear pair).

## Procedure
1. c_L = mean over paired samples of (x_old − x_new) at the chosen layer's input.
2. Fold: `ops.fold_input_map(W, consumer_w, np.eye(d), bias=consumer_b, c=c_L)` (b ← b + W c), equivalently
   `ops.compensate_input_shift(W, consumer_w, consumer_b, slice(None), -c_L)`.
3. Prefer the single layer with the largest f; avoid stacking offsets at many layers (they compound).

## Verification
Held-out fraction of the shift left (S07 metric) and chunk error vs the training-condition chunk; then rungs b–d.

## Expected effect & cost
Unknown. Seconds to compute; one bias vector per layer. Expected to be weaker than S07 for camera geometry changes
(30–60 % activation change at odd layers is unlikely to be a pure offset).

## Failure modes / interactions with other strategies
- An offset that is right on average is wrong per scene; check per-placement residuals.

## Evidence log
- (none yet)
