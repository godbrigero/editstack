"""EXAMPLE (ignored: names starting with '_' are not variants). Patterns that produced real wins.

1. Disconnect never-supervised padded inputs (exact algebra). Past model: action head padded 6 -> 32 channels,
   loss only on 6, but the 26 padding channels fed fresh noise back in at every denoising step -> jitter.
   Zeroing those input columns: jitter 4-6x lower, success 8/20 -> 17/20.
2. Fold a fitted linear map into a layer (closed form): features under measured cameras mapped back to the
   training-camera features -> in-distribution reach 6/40 -> 59/60 after a camera change.
3. Gated product circuit in two spare SwiGLU units: correction = (planned - current) * gain_readout, exactly zero
   while holding still, so in-distribution behaviour is untouched.
"""
import numpy as np

from editstack import ops

PARENT = None
OWNER = "example"


def apply(W, ctx):
    # 1. exact: disconnect padded input channels 6.. of the action input projection
    ops.zero_cols(W, "model.action_in_proj.weight", slice(6, None))

    # 2. closed form: M, c fitted offline with fit.paired_map(X_new, X_old) and saved to data/
    # M, c = ctx.data("cam_map.npz")["M"], ctx.data("cam_map.npz")["c"]
    # ops.fold_input_map(W, "model.expert.layers.1.self_attn.o_proj.weight", M)

    # 3. circuit: pick two unclaimed low-usage units of the last MLP and wire an exact product
    # L = "model.expert.layers.15.mlp."
    # pair = ops.free_units(ctx.stack, W, [(L + "gate_proj.weight", 0), (L + "up_proj.weight", 0),
    #                                       (L + "down_proj.weight", 1)], k=2)
    # a, b, out = ctx.data("pan_err.npy"), ctx.data("gate_feat.npy"), ctx.data("pan_out.npy")
    # ops.exact_product_pair(W, ctx, L + "gate_proj.weight", L + "up_proj.weight", L + "down_proj.weight",
    #                        pair, a, b, out, purpose="pan correction beyond trained band")
