"""Variants: a named chain of edits applied in memory to the untouched original weights.

edits/<name>/edit.py
    '''One paragraph: hypothesis, evidence (probe/run paths), expected effect, exactness argument.'''
    PARENT = "pad0"            # variant this builds on, or None for the original model
    OWNER = "approach"         # agent that owns it
    def apply(W, ctx):         # W: name -> float64 working copy on access; ctx: helpers (data, extra, claim, log)
        ...

Fitted numbers live in edits/<name>/data/ (npy/npz/json) and are loaded with ctx.data("file"), so an edit is a
deterministic function of (original weights, edit sources, data files). That function's hash is the variant's
`chain_hash`; runs record it, and the leaderboard ignores runs whose hash no longer matches.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import claims as claims_mod
from . import formats

ORIGINAL = "original"


class EditError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ loading edits


def edit_dir(stack, name: str) -> Path:
    return stack.edits_dir / name


def load_edit(stack, name: str):
    path = edit_dir(stack, name) / "edit.py"
    import sys
    env = str(stack.root / "env")
    if env not in sys.path:
        sys.path.insert(0, env)          # edits may import shared, exact helpers from env/
    if not path.exists():
        raise EditError(f"no edit named {name!r} ({path} missing)")
    spec = importlib.util.spec_from_file_location(f"edit_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "apply"):
        raise EditError(f"{path} must define apply(W, ctx)")
    return mod


ABLATE_SEP = "~minus~"


def read_parent(stack, name: str) -> Optional[str]:
    """PARENT of an edit, read from the source with ast (never executes the edit)."""
    import ast
    path = edit_dir(stack, name) / "edit.py"
    if not path.exists():
        raise EditError(f"no edit named {name!r} ({path} missing)")
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        raise EditError(f"{path}: syntax error: {e}")
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARENT" for t in node.targets):
            v = node.value
            if isinstance(v, ast.Constant) and (v.value is None or isinstance(v.value, str)):
                return v.value
            raise EditError(f"{path}: PARENT must be a string literal or None")
    return None


def split_ablation(name: str):
    """'v~minus~link' -> ('v', 'link'): the chain of v without one link (leave-one-out ablation)."""
    if ABLATE_SEP in name:
        base, link = name.split(ABLATE_SEP, 1)
        return base, link
    return name, None


def chain_of(stack, name: str) -> List[str]:
    """[root ... name] excluding 'original'. For 'v~minus~link' the chain of v with `link` removed."""
    base, drop = split_ablation(name)
    out, seen, cur = [], set(), base
    while cur and cur != ORIGINAL:
        if cur in seen:
            raise EditError(f"PARENT cycle at {cur}")
        seen.add(cur)
        out.append(cur)
        cur = read_parent(stack, cur)
    out = list(reversed(out))
    if drop is not None:
        if drop not in out:
            raise EditError(f"{drop} is not a link of {base} (chain: {out})")
        out = [v for v in out if v != drop]
    return out


def _env_fingerprint(stack) -> bytes:
    """Helper modules edits may import (env/*.py except the adapter) + the original weights' recorded hashes."""
    h = hashlib.sha256()
    env = stack.root / "env"
    for f in sorted(env.rglob("*")) if env.exists() else []:
        if f.is_file() and f.name != "adapter.py" and "__pycache__" not in f.parts and not f.name.endswith(".pyc"):
            h.update(f.relative_to(env).as_posix().encode())
            h.update(f.read_bytes())
    ref = stack.model_dir / "ORIGINAL.sha256.json"
    if ref.exists():
        h.update(ref.read_bytes())
    return h.digest()


def chain_hash(stack, name: str) -> str:
    """Identity of a variant's weights: every link's edit.py + data/, env helper modules, and the original's hashes.
    Changing any of these makes old runs of the variant stale."""
    h = hashlib.sha256(_env_fingerprint(stack))
    for v in chain_of(stack, name):
        d = edit_dir(stack, v)
        h.update(v.encode())
        files = [d / "edit.py"] + (sorted((d / "data").rglob("*")) if (d / "data").exists() else [])
        for f in files:
            if f.is_file():
                h.update(f.relative_to(d).as_posix().encode())
                h.update(f.read_bytes())
    return h.hexdigest()[:16]


def variant_exists(stack, name: str) -> bool:
    if name == ORIGINAL:
        return True
    try:
        chain_of(stack, name)
        return True
    except EditError:
        return False


# ------------------------------------------------------------------------------------------------ working weights


class Weights:
    """Dict-like view over the original tensors. Reading a name returns a float64 *working copy* that edits may
    modify in place; assigning replaces it (shape must match). Untouched tensors are never copied."""

    def __init__(self, ck: formats.Checkpoint):
        self._ck = ck
        self.work: Dict[str, np.ndarray] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self.work:
            if name not in self._ck.tensors:
                close = [n for n in self._ck.tensors if name.split(".")[-1] in n][:5]
                raise KeyError(f"no tensor {name!r}" + (f"; similar: {close}" if close else ""))
            self.work[name] = np.array(self._ck.tensors[name], dtype=np.float64)
        return self.work[name]

    def __setitem__(self, name: str, value) -> None:
        value = np.asarray(value, dtype=np.float64)
        orig = self._ck.tensors[name]
        if value.shape != orig.shape:
            raise EditError(f"{name}: shape {orig.shape} -> {value.shape} is not allowed (edits never change shapes)")
        self.work[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._ck.tensors

    def keys(self):
        return self._ck.tensors.keys()

    def original(self, name: str) -> np.ndarray:
        """Read-only original values (no copy)."""
        return self._ck.tensors[name]

    def shape(self, name: str):
        return self._ck.tensors[name].shape


@dataclass
class EditContext:
    stack: object
    variant: str
    owner: str = ""
    claims: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return edit_dir(self.stack, self.variant)

    def data(self, rel: str):
        """Load edits/<variant>/data/<rel> (.npy, .npz, .json, .txt)."""
        p = self.dir / "data" / rel
        if p.suffix == ".npy":
            return np.load(p, allow_pickle=False)
        if p.suffix == ".npz":
            return dict(np.load(p, allow_pickle=False))
        if p.suffix == ".json":
            return json.loads(p.read_text())
        return p.read_text()

    def extra(self, filename: str):
        """Load a non-weight file shipped with the original model (normalizer stats, config.json, ...)."""
        p = self.stack.original_dir / filename
        if p.suffix == ".json":
            return json.loads(p.read_text())
        if p.suffix == ".safetensors":
            ck = formats.Checkpoint(fmt="safetensors", source=p, tensors={}, dtypes={})
            formats._load_st(p, ck)
            return {k: np.array(v) for k, v in ck.tensors.items()}
        if p.suffix == ".npz":
            return dict(np.load(p, allow_pickle=False))
        return p.read_text()

    def claim(self, tensor: str, axis: int, indices, purpose: str = "") -> None:
        """Declare that this variant owns these indices of `tensor` along `axis` (e.g. spare MLP neurons)."""
        self.claims.append(claims_mod.make(tensor, axis, indices, self.variant, self.owner, purpose))

    def log(self, msg: str) -> None:
        self.notes.append(f"[{self.variant}] {msg}")


@dataclass
class Applied:
    name: str
    chain: List[str]
    chain_hash: str
    ck: formats.Checkpoint
    tensors: Dict[str, np.ndarray]           # full set, float32 (or native dtype) - ready for an adapter
    changed: List[str]
    stats: Dict[str, dict]
    claims: List[dict]
    notes: List[str]

    def storage(self, name: str) -> np.ndarray:
        return formats.to_storage(self.tensors[name], self.ck.dtypes[name])


def _delta_stats(orig: np.ndarray, new: np.ndarray) -> dict:
    d = new.astype(np.float64) - orig.astype(np.float64)
    nz = int(np.count_nonzero(d))
    on = float(np.linalg.norm(orig.astype(np.float64)))
    st = {"shape": list(orig.shape), "values_changed": nz, "frac_changed": round(nz / max(d.size, 1), 6),
          "max_abs_delta": float(np.abs(d).max()) if d.size else 0.0,
          "rel_fro_delta": float(np.linalg.norm(d) / on) if on > 0 else float("inf")}
    if d.ndim == 2 and min(d.shape) <= 4096 and nz:
        rows = np.flatnonzero(np.abs(d).sum(1))
        cols = np.flatnonzero(np.abs(d).sum(0))
        sub = d[np.ix_(rows, cols)]
        if min(sub.shape) <= 2048:
            s = np.linalg.svd(sub, compute_uv=False)
            st["rank"] = int((s > s[0] * 1e-6).sum()) if s.size and s[0] > 0 else 0
        st["rows_touched"], st["cols_touched"] = int(rows.size), int(cols.size)
    return st


def apply_variant(stack, name: str, check_claims: bool = True, ck: Optional[formats.Checkpoint] = None,
                  register: bool = False) -> Applied:
    """Apply the chain in memory. Claims are always checked against other variants' claims (declared AND derived:
    changing a slice someone else claimed is a conflict even if you didn't claim it); they are only written to
    model/CLAIMS.json when register=True (eval/export/promote), never by a dry `apply`."""
    ck = ck or stack.checkpoint()
    W = Weights(ck)
    chain = [] if name == ORIGINAL else chain_of(stack, name)
    all_claims: List[dict] = []
    notes: List[str] = []
    for v in chain:
        mod = load_edit(stack, v)
        ctx = EditContext(stack=stack, variant=v, owner=getattr(mod, "OWNER", ""))
        mod.apply(W, ctx)
        all_claims += ctx.claims
        notes += ctx.notes
    if check_claims:
        claims_mod.check(stack, all_claims, chain, W.work, ck)
        if register and all_claims:
            claims_mod.register(stack, all_claims, chain)
    tensors: Dict[str, np.ndarray] = {}
    changed, stats = [], {}
    for n, orig in ck.tensors.items():
        if n in W.work:
            new = W.work[n]
            if not np.all(np.isfinite(new)):
                raise EditError(f"{n}: edit produced NaN/Inf")
            stored = formats.to_storage(new, ck.dtypes[n])
            if not np.array_equal(stored, formats.to_storage(orig, ck.dtypes[n])):
                changed.append(n)
                view = formats.bf16_to_f32(stored) if ck.dtypes[n] == "BF16" else stored
                tensors[n] = view
                stats[n] = _delta_stats(orig, view)
                continue
        tensors[n] = orig
    return Applied(name=name, chain=chain, chain_hash=chain_hash(stack, name) if chain else "original", ck=ck,
                   tensors=tensors, changed=changed, stats=stats, claims=all_claims, notes=notes)


def diff_report(a: Applied) -> str:
    lines = [f"variant {a.name}   chain: {' -> '.join(['original', *a.chain])}   hash {a.chain_hash}",
             f"changed tensors: {len(a.changed)} of {len(a.tensors)}"]
    tot = 0
    for n in a.changed:
        s = a.stats[n]
        tot += s["values_changed"]
        extra = f" rank {s['rank']}" if "rank" in s else ""
        extra += f" rows {s['rows_touched']} cols {s['cols_touched']}" if "rows_touched" in s else ""
        lines.append(f"  {n:60s} {str(tuple(s['shape'])):>16s}  changed {s['values_changed']:>10d} "
                     f"({100 * s['frac_changed']:.3f}%)  max|d| {s['max_abs_delta']:.4g}  rel {s['rel_fro_delta']:.3g}{extra}")
    lines.append(f"total values changed: {tot}")
    for c in a.claims:
        lines.append(f"  claim: {c['tensor']} axis {c['axis']} idx {c['indices'][:12]}{'...' if len(c['indices']) > 12 else ''} - {c['purpose']}")
    lines += a.notes
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ export


def export(stack, name: str, out_dir: Optional[Path] = None, verify: bool = True) -> Path:
    """Bake a variant into a standalone checkpoint in the original format, then reload it and verify bit-exactness."""
    import shutil
    a = apply_variant(stack, name, register=True)
    out = Path(out_dir) if out_dir else stack.variants_dir / name
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    need = sum(p.stat().st_size for p in a.ck.weight_files + a.ck.extra_files)
    out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free
    if free < 1.2 * need + (1 << 30):
        raise EditError(f"not enough disk to export {name}: need ~{need / 2**30:.1f} GB (+1 GB margin), free {free / 2**30:.1f} GB")
    formats.save(a.ck, a.tensors, tmp)
    sums = tmp / "SHA256SUMS"
    if sums.exists():   # the original's checksum file would describe the ORIGINAL weights: regenerate it
        sums.write_text("".join(f"{formats.sha256_file(p)}  {p.name}\n" for p in sorted(tmp.iterdir())
                                if p.is_file() and p.name != "SHA256SUMS"))
    if verify:
        back = formats.load(tmp, a.ck.fmt)
        bad = [n for n in a.tensors
               if not np.array_equal(formats.to_storage(back.tensors[n], back.dtypes[n]), a.storage(n))]
        if bad:
            raise EditError(f"export verification failed for {len(bad)} tensors, e.g. {bad[:3]}")
    src = tmp / "EDIT_CHAIN"
    src.mkdir()
    for v in a.chain:
        shutil.copytree(edit_dir(stack, v), src / v, ignore=shutil.ignore_patterns("__pycache__", "runs", "*.pyc"))
    manifest = {
        "variant": name, "chain": a.chain, "chain_hash": a.chain_hash, "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": a.ck.fmt, "changed": {n: a.stats[n] for n in a.changed},
        "note": "Edits are already baked into the weights. EDIT_CHAIN/ is provenance only; never re-apply it.",
        "files": {p.name: formats.sha256_file(p) for p in sorted(tmp.iterdir()) if p.is_file()},
    }
    (tmp / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out
