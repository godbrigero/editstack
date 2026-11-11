# Edit cookbook

Convention: linear layer `y = x W^T + b`, `W` is `[out, in]`. All ops in `editstack.ops`, fits in `editstack.fit`.

## A. Exact algebra (preferred)

**Disconnect never-trained inputs.** Inputs that the loss never supervises but that are fed at inference (padding
channels, noise-only slots) inject noise. `ops.zero_cols(W, in_proj, slice(k, None))`. Exact on all trained
behaviour iff those channels carry no information (check the loss mask and data pipeline).

**Translate an input/output axis by delta (raw units).** Normalised delta `d = delta / std`.
- Consumer of the shifted quantity: `ops.compensate_input_shift(W, w, b, cols, d)` (b -= W[:, cols] d).
- Producer: `ops.shift_bias(W, b, idx, -d)` or `+d`, depending on direction.
- Generative/flow-matching heads: the noisy sample is `x_tau = tau*eps + (1-tau)*y`; translating y by d means the
  network sees `x - (1-tau) d` and must output `v - d`. The `(1-tau)` term needs a component linear in tau: fit
  coefficients c with `phi(tau) . c ~= tau` on the time features (exact at the sampling taus) and add
  `outer(W_x @ (A d), c)` to the time-feature columns of the first fused layer. (Worked example: SmolVLA
  `translate_actions`, docs in the original project edits.py.)

**Scale an axis by k (command gain without perception gain).** Output row `*k` (and bias), input column `/k`, and
fix the bias so a chosen centre C is a fixed point. Holding still is unchanged; every image-driven move is k x.
`ops.rescale_channel` is the function-preserving pair: run it FIRST and confirm outputs are unchanged, then remove
the compensating half on the side you want to change.

**Fold a linear map.** If the inputs a layer sees changed (new cameras, new normalisation), fit `x_old ~= M x_new + c`
on PAIRED samples (`fit.paired_map`), then `ops.fold_input_map(W, w, M, b, c)`. Do it at the layer(s) where the
mismatch is smallest and the map is most linear; check per-layer R^2 on held-out scenes AND placements.

## B. Closed-form fits

1. Capture X (layer input) on real rollouts of the parent variant (save during eval via ctx.out_dir, or
   adapter.capture on recorded observations). Group labels = episode/placement id.
2. Build target Y: the output the layer should have produced (e.g. current output + desired correction computed
   from the geometric error and the read-out direction).
3. `D, db = fit.fit_delta(X, Y, W, b, lam, rank=1..4)`; check `fit.cv_r2` with groups; save D to data/.
4. In edit.py: `ops.add_delta(W, name, ctx.data("D.npy"))`.
Rules: low rank; regularise; a fit that only works in-sample is not an edit.

## C. Gated circuits in spare units

- Units: `ops.free_units(ctx.stack, W, [(gate,0),(up,0),(down,1)], k)` picks the least-used unclaimed ones.
  Better: rank units by activation statistics from real rollouts and pass `score=`.
- Product: `ops.exact_product_pair` adds `(a.x)(b.x) * out` exactly (silu(z) - silu(-z) = z).
- Gate so the circuit is exactly 0 where the model already works: one factor = planned - current state (0 while
  holding), or a feature that is 0 inside the training envelope.
- Read-outs a, b: ridge fits from the MLP input (after norm) to the quantity you need (cube offset, error); check R^2.
- Claim every unit. Document the gate and the expected magnitude in the docstring.

## D. Verification recipes

- `editstack apply <v>`: only intended tensors change; rank as expected; claims listed.
- Offline equivalence: function-preserving half reproduces original outputs on recorded observations (max abs
  diff ~ float tolerance).
- Offline effect: the target output moves by the predicted amount on recorded observations.
- Only then: screen -> qualify -> heldout.
