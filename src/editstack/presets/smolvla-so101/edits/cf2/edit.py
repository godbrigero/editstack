"""Camera transfer: map expert attention outputs under MEASURED cameras back to the training-camera features.

Hypothesis: the real (measured) cameras differ in pose/FOV from the training cameras; perception features shift, so
every downstream circuit misfires (p17gshr: ID REACH 6/40 under measured cameras).
Mechanism: per-layer linear maps (ridge, per-layer CV lambda) fit on paired renders of the same states under both
camera sets, folded into all 16 expert attention outputs (camfix cf.apply_maps). Upstream fold => the circuits'
fitted inputs stay valid.
Evidence: cf2 near REACH 35/36, PICK 27/36; ID REACH 58/60, PICK 48/60 (measured cameras, 3 seeds each).
Caveat: maps fit partly on states from the same placements as the eval suites (cross-validated by scene, not
placement) - the held-out suite is mandatory before promotion.
DATA: data/maps2.npz (59 MB, not shipped in the public preset: copy agents/camfix/data/maps2.npz from the original
project into edits/cf2/data/).
"""
import legacy_bridge as lb

PARENT = "gshr"
OWNER = "camfix"


def apply(W, ctx):
    import cf
    lb.run_legacy(W, lambda sd: cf.apply_maps(sd, str(ctx.dir / "data" / "maps2.npz")))
