"""Stack adapter: the ONLY glue between editstack and this model's runtime + simulation.

editstack calls these functions; everything model- or sim-specific lives here (owned by the lead, Phase 0).
Implement load_policy and run_episode; episodes() and capture() are optional.

Contract
--------
load_policy(tensors, ctx) -> policy
    tensors: dict name -> numpy array (the variant's full weights, same names/shapes as weights/original;
             bf16 tensors arrive as float32 - cast back to the module's dtype when loading).
    ctx.original_dir: weights/original (configs, tokenizers, normaliser stats live there)
    ctx.device: from stack.json "device" or --device
    Build the model ONCE per process; it is reused for every episode.

run_episode(policy, episode, seed, ctx) -> dict
    episode: one item of the suite's "episodes" list (e.g. {"x": 0.1, "y": -0.2, "yaw": 0.3})
    seed:    policy-noise seed. Seed the model's sampling noise from (seed, episode) so that runs are
             deterministic and comparisons between variants are PAIRED.
    ctx.out_dir: per-episode artifact folder (save trajectories/videos here if useful; create it).
    Must return {"metrics": {<every name in stack.json metrics.all>: bool}, "outcome": "<short code>", ...}
    Extra keys (errors in cm, timings) are kept in results.jsonl for probes.

episodes(suite) -> list[dict]       optional: generate a suite's episodes when the json has no "episodes" list
capture(policy, obs, names) -> dict optional: activations at named modules for probes (forward hooks)

Checklist before Phase 1 (see docs/HARNESS_AUDIT.md): deterministic, sim cameras == real cameras, same
preprocessing/normalisation as deployment, same chunk execution as deployment, staged metrics + outcome codes.
"""
from __future__ import annotations

from pathlib import Path


def load_policy(tensors, ctx):
    raise NotImplementedError("Phase 0: implement env/adapter.py load_policy (see docstring)")


def run_episode(policy, episode, seed, ctx):
    raise NotImplementedError("Phase 0: implement env/adapter.py run_episode (see docstring)")
