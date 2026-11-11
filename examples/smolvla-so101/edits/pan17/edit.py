"""Pan (joint 0) command gain x1.7 with perception compensated: generalises reach to cubes 2-6 cm outside the band.

Hypothesis: beyond the trained pan range the policy's commands compress toward the hold; scaling the pan axis
(output x k, state input / k, centred on the action mean) leaves holding unchanged and makes every image-driven
pan move 1.7x larger.
Evidence: near-lateral REACH 5/12 (pad0_deep8) -> 31/36 (3 seeds); ID cost 79/80 -> 76/80 reach (training cameras).
Exactness: affine reparameterisation of one action axis; verified offline (max diff 0.05 raw units) before sim.
"""
import smolvla_algebra as sa

PARENT = "deep8"
OWNER = "lead"


def apply(W, ctx):
    sa.scale_axis(W, 0, 1.7, sa.stats(ctx))
