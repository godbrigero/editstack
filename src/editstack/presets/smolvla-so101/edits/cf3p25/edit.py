"""cf2 refit with extra misaligned low-hover captures + height-gated pan servo gain (camfix cf3p25).

Evidence: near REACH 34/36, PICK 29/36; ID REACH 59/60, PICK 51/60 (measured cameras). More picks than cf2, one
fewer near REACH - the two are within noise of each other on REACH; prefer by PICK if REACH ties on held-out.
DATA: data/maps3.npz (59 MB, not shipped: copy agents/camfix/data/maps3.npz from the original project).
"""
import legacy_bridge as lb

PARENT = "gshr"
OWNER = "camfix"
L15 = "model.vlm_with_expert.lm_expert.layers.15.mlp."


def apply(W, ctx):
    import cf
    st = lb.stats_np32(ctx)

    def f(sd):
        cf.apply_maps(sd, str(ctx.dir / "data" / "maps3.npz"))
        cf.low_pan_gain(sd, st["astd"], beta=0.25)
    lb.run_legacy(W, f)
    for t, ax in (("gate_proj.weight", 0), ("up_proj.weight", 0), ("down_proj.weight", 1)):
        ctx.claim(L15 + t, ax, [301, 1703], "low-hover pan gain")
