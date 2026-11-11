"""Registry of which tensor slices (neurons, channels, rows, columns) each variant writes to.

Two agents writing the same spare neuron was a real failure in past runs (approach and grasp both picked neurons 779
and 1893). Every slice-level edit that *adds* a mechanism must claim its indices; applying a variant whose claims
overlap a claim held by a variant outside its own parent chain is an error.

model/CLAIMS.json: [{"tensor", "axis", "indices", "variant", "owner", "purpose", "date"}]
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, List, Sequence


def _path(stack) -> Path:
    return stack.model_dir / "CLAIMS.json"


def load(stack) -> List[dict]:
    p = _path(stack)
    return json.loads(p.read_text()) if p.exists() else []


def save(stack, claims: List[dict]) -> None:
    p = _path(stack)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(claims, indent=1) + "\n")


def normalize(indices) -> List[int]:
    if isinstance(indices, slice):
        raise ValueError("claim explicit indices (or range(...)), not a slice")
    if isinstance(indices, int):
        return [indices]
    return sorted({int(i) for i in indices})


def conflicts(existing: Iterable[dict], new: dict, allowed_variants: Sequence[str]) -> List[dict]:
    out = []
    ni = set(new["indices"])
    for c in existing:
        if c["tensor"] != new["tensor"] or c["axis"] != new["axis"] or c["variant"] in allowed_variants:
            continue
        overlap = ni & set(c["indices"])
        if overlap:
            out.append({**c, "overlap": sorted(overlap)[:20]})
    return out


def _lock(stack):
    import fcntl
    import os
    p = _path(stack).with_suffix(".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _unlock(fd) -> None:
    import fcntl
    import os
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _live(stack, existing: List[dict]) -> List[dict]:
    """Drop claims whose variant no longer exists (deleted edits must not block others forever)."""
    from .variant import variant_exists
    return [c for c in existing if variant_exists(stack, c["variant"])]


def touched(work: dict, ck, tensor: str, axis: int) -> set:
    """Indices along `axis` of `tensor` that this edit actually changed (derived from the diff)."""
    import numpy as np
    if tensor not in work:
        return set()
    new = np.asarray(work[tensor], dtype=np.float64)
    old = np.asarray(ck.tensors[tensor], dtype=np.float64)
    if new.ndim == 0:
        return {0} if new != old else set()
    d = np.abs(new - old)
    other = tuple(i for i in range(d.ndim) if i != axis)
    return {int(i) for i in np.flatnonzero(d.sum(axis=other) if other else d)}


def check(stack, new_claims: List[dict], chain: Sequence[str], work: dict = None, ck=None) -> None:
    """Raise ClaimConflict if this chain claims, or merely CHANGES, a slice claimed by a variant outside it."""
    existing = _live(stack, load(stack))
    lines = []
    for c in new_claims:
        for b in conflicts(existing, c, chain):
            lines.append(f"  {c['variant']} claims {c['tensor']} axis {c['axis']} idx {b['overlap']} "
                         f"already claimed by {b['variant']} ({b.get('owner')}: {b.get('purpose')})")
    if work is not None and ck is not None:
        for c in existing:
            if c["variant"] in chain:
                continue
            hit = touched(work, ck, c["tensor"], c["axis"]) & set(c["indices"])
            if hit:
                lines.append(f"  chain {'->'.join(chain)} CHANGES {c['tensor']} axis {c['axis']} idx {sorted(hit)[:20]} "
                             f"claimed by {c['variant']} ({c.get('owner')}: {c.get('purpose')}) without claiming it")
    if lines:
        raise ClaimConflict("claim conflict:\n" + "\n".join(lines) +
                            "\nPick other indices (editstack.ops.free_units) or build on that variant as PARENT.")


def register(stack, new_claims: List[dict], chain: Sequence[str]) -> None:
    """Record the claims of `chain` (under a file lock; prunes claims of deleted variants)."""
    if not new_claims:
        return
    fd = _lock(stack)
    try:
        existing = _live(stack, load(stack))
        for c in new_claims:
            if conflicts(existing, c, chain):
                raise ClaimConflict(f"claim conflict while registering {c['variant']} (re-run apply to see it)")
        keyed = {(c["tensor"], c["axis"], c["variant"], tuple(c["indices"])) for c in existing}
        for c in new_claims:
            k = (c["tensor"], c["axis"], c["variant"], tuple(c["indices"]))
            if k not in keyed:
                existing.append(c)
                keyed.add(k)
        save(stack, existing)
    finally:
        _unlock(fd)


def make(tensor: str, axis: int, indices, variant: str, owner: str = "", purpose: str = "") -> dict:
    return {"tensor": tensor, "axis": int(axis), "indices": normalize(indices), "variant": variant,
            "owner": owner, "purpose": purpose, "date": time.strftime("%Y-%m-%d %H:%M")}


def claimed_indices(stack, tensor: str, axis: int) -> set:
    return {i for c in load(stack) if c["tensor"] == tensor and c["axis"] == axis for i in c["indices"]}


class ClaimConflict(RuntimeError):
    pass
