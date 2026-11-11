"""Run the original project's torch edit functions (env/legacy/*.py) on editstack working weights.

The legacy functions take a state dict `sd` of torch tensors in the checkpoint's STORAGE dtypes (bf16 for the
expert layers, float32 elsewhere) and edit it in place or by reassignment. run_legacy(W, fn) emulates exactly that:
each tensor is handed over in its storage dtype (so intermediate bf16 rounding happens as it did in the old export
pipeline), and every tensor the function touched is written back into W afterwards.
"""
import sys
from pathlib import Path

import numpy as np

LEGACY = Path(__file__).resolve().parent / "legacy"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))


class TorchSD(dict):
    def __init__(self, W):
        super().__init__()
        self.W = W

    def _dtype(self, k):
        import torch
        return torch.bfloat16 if self.W._ck.dtypes.get(k) == "BF16" else torch.float32

    def __getitem__(self, k):
        import torch
        if not dict.__contains__(self, k):
            t = torch.from_numpy(np.asarray(self.W.original(k), np.float32).copy()) if k not in self.W.work \
                else torch.from_numpy(self.W[k].astype(np.float32))
            dict.__setitem__(self, k, t.to(self._dtype(k)))
        return dict.__getitem__(self, k)

    def __setitem__(self, k, v):
        dict.__setitem__(self, k, v)

    def __contains__(self, k):
        return k in self.W

    def keys(self):
        return self.W.keys()


def run_legacy(W, fn):
    """fn(sd) with sd a TorchSD over W; touched tensors are written back (float64 copies of the torch results)."""
    import torch
    sd = TorchSD(W)
    fn(sd)
    for k in list(dict.keys(sd)):
        t = dict.__getitem__(sd, k)
        new = t.detach().to(torch.float32).cpu().numpy().astype(np.float64)
        if k in W.work or not np.array_equal(new, np.asarray(W.original(k), np.float64)):
            W[k] = new


def stats_np32(ctx):
    """AMEAN/ASTD/SMEAN/SSTD exactly as the old pipeline passed them (float32 numpy)."""
    import smolvla_algebra as sa
    st = sa.stats(ctx)
    return {k: v.astype(np.float32) for k, v in st.items()}
