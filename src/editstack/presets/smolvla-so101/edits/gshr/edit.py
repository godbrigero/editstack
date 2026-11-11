"""Gated approach circuits + gated roll servo (approach agent's p17gshr minus its pad0/deep8/pan17 prefix).

Hypothesis: beyond the trained band, pan17's gain over-aims in-band (1.5x) and the image servo is weak; wrist roll
is never re-aimed (49/69 reached-not-picked had roll > 25 deg off).
Mechanism: exact SwiGLU hinge/product pairs in least-used expert layer-15 units, gated on cube-x (out-of-band) and
on gripper state; silent while holding in-band. Readouts: data in env/legacy/data (readout_L15_p17, grs_readout_v2).
Evidence (training cameras): p17gshr near REACH 35/36, PICK 32/36; ID REACH 40/40, PICK 34/40 (3+2 seeds).
Roll part: alignment improved, pick gain within noise (independent review). Under MEASURED cameras this chain needs
cf2/cf3p25 on top (p17gshr alone: ID REACH 6/40).
"""
import legacy_bridge as lb

PARENT = "pan17"
OWNER = "approach+grasp"
L15 = "model.vlm_with_expert.lm_expert.layers.15.mlp."
NEURONS = {"gated_gain": [1110, 1985, 177, 1307, 1155, 336], "gated_servo": [2015, 1263, 573, 170],
           "gated_hold": [57, 1701, 354, 1488, 248, 2008], "gated_roll2": [779, 1893]}


def apply(W, ctx):
    import gpg, grs
    st = lb.stats_np32(ctx)
    RO = str(lb.LEGACY / "data" / "readout_L15_p17.npz")

    def f(sd):
        gpg.gated_gain(sd, st["astd"], c=0.41, g0=0.065, g1=0.095, readout=RO, neurons=NEURONS["gated_gain"])
        gpg.gated_servo(sd, st["astd"], g0=0.065, g1=0.095, readout=RO, neurons=NEURONS["gated_servo"])
        gpg.gated_hold(sd, st["astd"], gamma=0.08, g0=0.065, g1=0.095, readout=RO, neurons=NEURONS["gated_hold"])
        grs.gated_roll2(sd, 0.020)
    lb.run_legacy(W, f)
    for purpose, idx in NEURONS.items():
        for t, ax in (("gate_proj.weight", 0), ("up_proj.weight", 0), ("down_proj.weight", 1)):
            ctx.claim(L15 + t, ax, idx, purpose)
