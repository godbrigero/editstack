"""Toy adapter: 'reach' a 2-D target. success = |policy(target) - target| < 0.08."""
import numpy as np


def load_policy(tensors, ctx):
    return {k: np.asarray(v, np.float64) for k, v in tensors.items()}


def run_episode(policy, episode, seed, ctx):
    rng = np.random.default_rng(seed * 100003 + int(episode["id"]))
    target = np.array([episode["x"], episode["y"]])
    inp = np.r_[target, rng.normal(0, 1.0, 6)]                  # pad channels get noise at inference
    h = np.maximum(policy["enc.weight"] @ inp + policy["enc.bias"], 0)
    out = policy["head.weight"] @ h + policy["head.bias"]
    err = float(np.linalg.norm(out - target))
    return {"metrics": {"success": err < 0.08}, "outcome": "ok" if err < 0.08 else "miss", "err": round(err, 4)}
