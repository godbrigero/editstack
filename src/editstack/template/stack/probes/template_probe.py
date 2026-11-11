"""Template: offline probe on recorded observations (seconds per question, no sim).

Pattern that located every past fix:
  1. load the policy for a variant (takes a RAM slot)
  2. feed recorded observations (saved by run_episode into runs/<v>/<suite>/s<k>/ep<i>/)
  3. vary ONE thing (an input channel, a weight, a camera image) and measure ONE number at the output
  4. print it, write it to MODEL_MAP.md with this script's path as evidence
"""
import argparse

import numpy as np  # noqa: F401

from editstack import Stack, fit  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--variant", default="original")
a = p.parse_args()

st = Stack.find()
policy = st.policy(a.variant)            # RAM slot + in-memory edit chain + env/adapter.load_policy
adapter = st.adapter()
# obs = np.load("runs/original/id/s0/ep0/obs_000.npz")
# acts = adapter.capture(policy, obs, ["model.expert.layers.15.mlp"])   # if implemented
# print(fit.cv_r2(X, y, groups=episode_ids))
print("template probe: fill in")
