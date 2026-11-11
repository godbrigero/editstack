"""{desc}

Hypothesis:   (what is wrong, where, with the number that shows it)
Evidence:     (probe output / run dirs / MODEL_MAP lines)
Expected:     (the metric/outcome change you predict, with a number)
Exactness:    (why everything else is unchanged: algebraic identity, gating, rank, claimed units)
"""
import numpy as np

from editstack import ops, fit  # noqa: F401

PARENT = {parent}          # None = builds directly on the original model
OWNER = "{owner}"


def apply(W, ctx):
    # W["tensor.name"] -> float64 working copy (modify in place, or assign a same-shape array)
    # ctx.data("x.npy") -> edits/{name}/data/x.npy ; ctx.extra("config.json") -> file from weights/original
    # ctx.claim(tensor, axis, indices, purpose) for any unit you take over ; ctx.log("...")
    raise NotImplementedError("write the edit")
