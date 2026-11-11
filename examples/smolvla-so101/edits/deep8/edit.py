"""Seat the grasp ~8 mm deeper: exact translation of the action (and state) space by a Jacobian step.

Hypothesis: the gripper closes 6-10 mm short on the jaw tips (geo.py). MuJoCo Jacobian: 8 mm deeper =
joint delta [0, 1.52, 1.52, -3.11, 0, 0] (raw units).
Evidence: pad0_deep8 19/20 held >= 1 s vs 17/20 for pad0 (single seed).
Exactness: flow-matching identity: x_tau = tau eps + (1-tau)(y + d) -> bias shifts in action_in_proj /
action_out_proj / state_proj plus a tau-linear term through the time features (fit error ~1e-6 at the 10 taus).
"""
import smolvla_algebra as sa

PARENT = "pad0"
OWNER = "lead"


def apply(W, ctx):
    err = sa.translate_actions(W, [0, 1.52, 1.52, -3.11, 0, 0], sa.stats(ctx))
    ctx.log(f"tau fit max error at inference taus: {err:.2e}")
