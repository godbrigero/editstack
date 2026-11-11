"""Build a toy 'policy' with a real-world-style defect: padded input channels that were never trained.

policy(obs) = W2 @ relu(W1 @ inp + b1) + b2, inp = [x, y, pad0..pad5]. At inference the 6 pad channels carry noise
(like the padded action channels of a flow-matching head). W1's pad columns are still at their random init.
Usage: python make_model.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from editstack.formats import write_safetensors  # noqa: E402

rng = np.random.default_rng(0)
H = 64
W1 = np.zeros((H, 8))
W1[:, :2] = rng.normal(0, 2.0, (H, 2))                         # "trained" real-input columns
W1[:, 2:] = rng.uniform(-1 / np.sqrt(8), 1 / np.sqrt(8), (H, 6))  # pad columns: untouched init
b1 = rng.normal(0, 0.5, H)
# fit the read-out on clean in-distribution data (|x|,|y| <= 0.5), pad inputs = 0 during "training"
X = rng.uniform(-0.5, 0.5, (4000, 2))
Hd = np.maximum(np.c_[X, np.zeros((4000, 6))] @ W1.T + b1, 0)
A = np.c_[Hd, np.ones(4000)]
sol = np.linalg.lstsq(A, X, rcond=None)[0]
W2, b2 = sol[:-1].T, sol[-1]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
t = {"enc.weight": W1, "enc.bias": b1, "head.weight": W2, "head.bias": b2}
write_safetensors(out / "model.safetensors", {k: v.astype(np.float32) for k, v in t.items()},
                  {"enc.weight": "BF16", "enc.bias": "F32", "head.weight": "F32", "head.bias": "F32"})
(out / "config.json").write_text('{"hidden": 64, "pad_channels": 6}\n')
print("wrote", out)
