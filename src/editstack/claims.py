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


def register(stack, new_claims: List[dict], chain: Sequence[str]) -> None:
    """Check and record the claims made while applying `chain` (last = the variant itself)."""
    if not new_claims:
        return
    existing = load(stack)
    problems = []
    for c in new_claims:
        bad = conflicts(existing, c, chain)
        if bad:
            problems.append((c, bad))
    if problems:
        lines = []
        for c, bad in problems:
            for b in bad:
                lines.append(f"  {c['variant']} wants {c['tensor']} axis {c['axis']} idx {b['overlap']} "
                             f"already claimed by {b['variant']} ({b.get('owner')}: {b.get('purpose')})")
        raise ClaimConflict("claim conflict:\n" + "\n".join(lines) +
                            "\nPick other indices (editstack.ops.free_units) or build on that variant as PARENT.")
    keyed = {(c["tensor"], c["axis"], c["variant"], tuple(c["indices"])) for c in existing}
    changed = False
    for c in new_claims:
        k = (c["tensor"], c["axis"], c["variant"], tuple(c["indices"]))
        if k not in keyed:
            existing.append(c)
            keyed.add(k)
            changed = True
    if changed:
        save(stack, existing)


def make(tensor: str, axis: int, indices, variant: str, owner: str = "", purpose: str = "") -> dict:
    return {"tensor": tensor, "axis": int(axis), "indices": normalize(indices), "variant": variant,
            "owner": owner, "purpose": purpose, "date": time.strftime("%Y-%m-%d %H:%M")}


def claimed_indices(stack, tensor: str, axis: int) -> set:
    return {i for c in load(stack) if c["tensor"] == tensor and c["axis"] == axis for i in c["indices"]}


class ClaimConflict(RuntimeError):
    pass
