"""Disconnect the 26 padded action-input channels (6..31) of action_in_proj.

Hypothesis: the loss is computed on channels 0-5 only (losses[:, :, :6]); channels 6-31 are never supervised but
are fed back as fresh noise at each of the 10 denoising steps -> in-chunk jitter.
Evidence: decode SCAN.md flags action_in_proj cols [0:6] vs [6:32]; original project: jitter 0.91 -> 0.19,
held >= 1 s 8/20 -> 17/20 (smolvla-weight-edit-6000, 2026-11-05 23:24 UTC).
Exactness: changes only how noise in unsupervised channels enters; supervised channels untouched.
"""
import smolvla_algebra as sa

PARENT = None
OWNER = "lead"


def apply(W, ctx):
    sa.pad_zero(W)
