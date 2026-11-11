"""`editstack decode`: turn a pile of weights into facts an agent can start from.

Writes into model/:
  INVENTORY.md / inventory.json   every tensor: shape, dtype, stats; repeated blocks collapsed; role guesses
  SCAN.md                          structural-defect hints (padding/untrained slices, dead units, tied tensors)
  ORIGINAL.sha256.json             hashes of the original files (the original must never change)
  ARCHITECTURE.md                  created once from a template: the agent must fill and VERIFY it from code

Everything here is computed from weights only, so it is HINTS. Facts about the forward pass (what is padded, which
loss terms are masked, normalisation, the inference loop) must come from reading the model/training code.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import formats

ROLE_KEYWORDS = [
    ("attention.q", r"(^|[._])(q_proj|query|wq|to_q)([._]|$)"),
    ("attention.k", r"(^|[._])(k_proj|key|wk|to_k)([._]|$)"),
    ("attention.v", r"(^|[._])(v_proj|value|wv|to_v)([._]|$)"),
    ("attention.qkv", r"(qkv|in_proj_weight|c_attn)"),
    ("attention.out", r"(o_proj|attn\w*\.out_proj|self_attn\.out|(^|[._])wo([._]|$)|to_out|attn\.proj|c_proj|attention\.output\.dense)"),
    ("mlp.gate", r"(gate_proj|w1|gate)"),
    ("mlp.up", r"(up_proj|w3|fc1|c_fc|intermediate\.dense|mlp\.0|lin1)"),
    ("mlp.down", r"(down_proj|w2|fc2|mlp\.2|output\.dense|lin2)"),
    ("norm", r"(norm|ln_|layernorm|\.ln\d?\.|rms)"),
    ("embedding", r"(embed|wte|tok_|token_emb|word_emb)"),
    ("position", r"(pos_emb|position|wpe|rope|rotary)"),
    ("patch/conv", r"(patch|conv)"),
    ("head/projection", r"(lm_head|head|proj$|projection|readout|action_|state_|out_proj)"),
]


def _role(name: str) -> str:
    low = name.lower()
    for role, pat in ROLE_KEYWORDS:
        if re.search(pat, low):
            return role
    return "?"


def _pattern(name: str) -> str:
    return re.sub(r"(?<=[._])\d+(?=[._]|$)", "{i}", name)


def _sample(a: np.ndarray, k: int = 1_000_000) -> np.ndarray:
    f = np.asarray(a).reshape(-1)
    if f.size <= k:
        return f.astype(np.float64)
    idx = np.random.default_rng(0).choice(f.size, k, replace=False)
    return f[np.sort(idx)].astype(np.float64)


def _kurt(x: np.ndarray) -> float:
    x = x - x.mean()
    v = (x * x).mean()
    return float((x ** 4).mean() / (v * v)) if v > 0 else float("nan")


def tensor_stats(a: np.ndarray) -> dict:
    if a.dtype.kind not in "fiu" or a.size == 0:
        return {}
    s = _sample(a)
    fin = np.isfinite(s)
    s = s[fin]
    out = {"mean": float(s.mean()) if s.size else 0.0, "std": float(s.std()) if s.size else 0.0,
           "absmax": float(np.abs(s).max()) if s.size else 0.0, "frac_zero": float((s == 0).mean()) if s.size else 0.0,
           "nonfinite_in_sample": int((~fin).sum())}
    if s.size > 16 and a.dtype.kind == "f":
        out["kurtosis"] = round(_kurt(s), 3)
    return out


# ----------------------------------------------------------------------------------------------- structural scan


def _slice_scan(name: str, a: np.ndarray) -> List[str]:
    """Hints for 2-D weights along every small axis (<=256): zero/dead slices, init-like slices, head/tail splits."""
    hints = []
    if a.ndim != 2 or a.dtype.kind != "f":
        return hints
    a = np.asarray(a, dtype=np.float64)
    for axis in (0, 1):
        n = a.shape[axis]
        if n > 256 or n < 4:
            continue
        what = "rows (outputs)" if axis == 0 else "cols (inputs)"
        sl = a if axis == 0 else a.T                        # [n, m]
        norms = np.linalg.norm(sl, axis=1)
        zero = np.flatnonzero(norms == 0)
        if zero.size:
            hints.append(f"{name}: {zero.size} all-zero {what}: {_ranges(zero)}")
        kur = np.array([_kurt(r) if r.std() > 0 else np.nan for r in sl])
        ln = np.log(np.maximum(norms, 1e-12))
        # best contiguous split head|tail by norm or kurtosis
        best = None
        for k in range(2, n - 1):
            for stat, v in (("norm", ln), ("kurtosis", kur)):
                h, t = v[:k], v[k:]
                h, t = h[np.isfinite(h)], t[np.isfinite(t)]
                if h.size < 2 or t.size < 2:
                    continue
                pooled = np.sqrt((h.var() * h.size + t.var() * t.size) / (h.size + t.size)) + 1e-9
                eff = abs(h.mean() - t.mean()) / pooled
                if best is None or eff > best[0]:
                    best = (eff, k, stat, h.mean(), t.mean())
        if best and best[0] >= 3.0:
            eff, k, stat, hm, tm = best
            if stat == "norm":
                hm, tm = float(np.exp(hm)), float(np.exp(tm))
            hints.append(f"{name}: {what} split at {k}: [0:{k}] vs [{k}:{n}] differ in {stat} "
                         f"({hm:.3g} vs {tm:.3g}, effect {eff:.1f}). Possible padding / unsupervised channels "
                         f"- check the training loss mask and how these channels are fed at inference.")
        uni = np.flatnonzero(np.isfinite(kur) & (kur < 2.0))
        if 0 < uni.size < n and uni.size >= 2 and np.nanmedian(kur) > 2.4:
            hints.append(f"{name}: {uni.size}/{n} {what} have uniform-like kurtosis (<2.0; init-like, maybe never "
                         f"trained) while the median is {np.nanmedian(kur):.2f}: {_ranges(uni)}")
    return hints


def _ranges(idx) -> str:
    idx = sorted(int(i) for i in idx)
    out, start, prev = [], None, None
    for i in idx:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = i
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
    s = ",".join(out)
    return s if len(s) < 200 else s[:200] + "..."


def scan(ck: formats.Checkpoint, stats: Dict[str, dict]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = OrderedDict(nonfinite=[], constant=[], tied=[], uniform_like=[], slices=[], io_dims=[])
    digests = defaultdict(list)
    for n, a in ck.tensors.items():
        st = stats.get(n, {})
        if st.get("nonfinite_in_sample"):
            out["nonfinite"].append(f"{n}: NaN/Inf present")
        if a.size > 1 and st and st["std"] == 0:
            out["constant"].append(f"{n}: constant value {st['mean']:.4g} ({'all zero' if st['mean'] == 0 else 'constant'})")
        if a.size >= 1024:
            import hashlib
            digests[(a.shape, hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest())].append(n)
        if st.get("kurtosis") is not None and st["kurtosis"] < 2.0 and a.ndim == 2 and a.size >= 256:
            out["uniform_like"].append(f"{n}: whole-tensor kurtosis {st['kurtosis']} (uniform init is 1.8) - "
                                       f"possibly barely trained or unused")
        out["slices"] += _slice_scan(n, a)
        if a.ndim == 2 and min(a.shape) <= 64 and max(a.shape) >= 4 * min(a.shape):
            out["io_dims"].append(f"{n} {tuple(a.shape)}: small side {min(a.shape)} - a model input/output interface? "
                                  f"Find which of its {min(a.shape)} channels carry real data.")
    for (shape, _), names in digests.items():
        if len(names) > 1:
            out["tied"].append(f"identical tensors {shape}: {names}")
    return {k: _collapse(v, list(ck.tensors)) for k, v in out.items()}


def _collapse(items: List[str], names: List[str]) -> List[str]:
    """Merge hints that are identical except for the layer index: 'a.{i}.b: msg (16x, i=0..15)'."""
    groups: Dict[tuple, List[int]] = OrderedDict()
    loose = []
    name_set = set(names)
    for h in items:
        n = h.split(":", 1)[0].split(" ", 1)[0]
        if n not in name_set:
            loose.append(h)
            continue
        pat = _pattern(n)
        nums = re.findall(r"(?<=[._])(\d+)(?=[._]|$)", n)
        groups.setdefault((pat, h[len(n):]), []).extend(int(x) for x in nums[:1])
    out = []
    for (pat, rest), idx in groups.items():
        if len(idx) > 1:
            out.append(f"{pat}{rest} ({len(idx)}x, i={_ranges(idx)})")
        else:
            out.append(f"{pat.replace('{i}', str(idx[0])) if idx else pat}{rest}")
    return out + loose


# ----------------------------------------------------------------------------------------------- reports


def decode(stack, write: bool = True) -> dict:
    ck = stack.checkpoint()
    stats = {n: tensor_stats(a) for n, a in ck.tensors.items()}
    groups: Dict[str, List[str]] = OrderedDict()
    for n in ck.tensors:
        groups.setdefault(_pattern(n), []).append(n)
    anonymous = sum(bool(re.fullmatch(r"[A-Za-z]{0,3}\d+(\.(weight|bias))?", n)) for n in ck.tensors) > 0.5 * len(ck.tensors)
    total = int(sum(a.size for a in ck.tensors.values()))
    inv = {
        "format": ck.fmt, "source": str(ck.source), "files": [p.name for p in ck.weight_files],
        "extra_files": [p.name for p in ck.extra_files], "n_tensors": len(ck.tensors), "n_params": total,
        "dtypes": dict(sorted(_count(ck.dtypes.values()).items())), "anonymous_names": anonymous,
        "tensors": {n: {"shape": list(a.shape), "dtype": ck.dtypes[n], "params": int(a.size), "role": _role(n),
                        "group": _pattern(n), **stats[n]} for n, a in ck.tensors.items()},
    }
    hints = scan(ck, stats)
    if not write:
        return {"inventory": inv, "hints": hints}
    md = stack.model_dir
    md.mkdir(parents=True, exist_ok=True)
    (md / "inventory.json").write_text(json.dumps(inv, indent=1) + "\n")
    (md / "INVENTORY.md").write_text(_inventory_md(inv, groups, ck))
    (md / "SCAN.md").write_text(_scan_md(hints))
    hashes = {p.name: formats.sha256_file(p) for p in ck.weight_files}
    (md / "ORIGINAL.sha256.json").write_text(json.dumps(hashes, indent=1) + "\n")
    arch = md / "ARCHITECTURE.md"
    if not arch.exists() or "AUTO-FACTS" in arch.read_text()[:400]:
        arch.write_text(_arch_md(stack, inv, groups))
    if stack.cfg["weights"].get("format") != ck.fmt:
        stack.cfg["weights"]["format"] = ck.fmt
        stack.save_cfg()
    return {"inventory": inv, "hints": hints}


def _count(it):
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return d


def _fmt_params(n: int) -> str:
    for unit, k in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= k:
            return f"{n / k:.2f}{unit}"
    return str(n)


def _inventory_md(inv, groups, ck) -> str:
    L = [f"# Weight inventory (auto-generated by `editstack decode`; do not edit)\n",
         f"- format: **{inv['format']}**, files: {', '.join(inv['files'])}",
         f"- extra files carried with the model: {', '.join(inv['extra_files']) or '-'}",
         f"- tensors: {inv['n_tensors']}, parameters: {_fmt_params(inv['n_params'])} ({inv['n_params']:,})",
         f"- storage dtypes: {inv['dtypes']}"]
    if inv["anonymous_names"]:
        L.append("- **names are anonymous** (e.g. L1, L2...): roles cannot be guessed from names. Group by shape "
                 "below and recover the wiring from the model description/code. Record it in ARCHITECTURE.md.")
    # top-level tree
    tree = defaultdict(int)
    for n, t in inv["tensors"].items():
        parts = n.split(".")
        for d in range(1, min(3, len(parts)) + 1):
            tree[".".join(parts[:d])] += t["params"]
    L.append("\n## Parameter tree (top 3 levels)\n")
    for k in sorted(tree):
        depth = k.count(".")
        if tree[k] >= max(1, inv["n_params"] // 2000) or depth == 0:
            L.append(f"{'  ' * depth}- `{k}` {_fmt_params(tree[k])}")
    L.append("\n## Groups (repeated blocks collapsed; {i} = layer index)\n")
    L.append("| pattern | count | shape | dtype | role guess | std (first) | kurtosis (first) |")
    L.append("|---|---|---|---|---|---|---|")
    for pat, names in groups.items():
        t = inv["tensors"][names[0]]
        idx = ""
        if len(names) > 1:
            nums = sorted({int(m) for n in names for m in re.findall(r"(?<=[._])(\d+)(?=[._]|$)", n)})
            idx = f" (i={nums[0]}..{nums[-1]})" if nums else ""
        L.append(f"| `{pat}` | {len(names)}{idx} | {tuple(t['shape'])} | {t['dtype']} | {t['role']} | "
                 f"{t.get('std', 0):.3g} | {t.get('kurtosis', '-')} |")
    if inv["anonymous_names"]:
        by_shape = defaultdict(list)
        for n, t in inv["tensors"].items():
            by_shape[tuple(t["shape"])].append(n)
        L.append("\n## Anonymous names grouped by shape\n")
        for shp, names in sorted(by_shape.items(), key=lambda kv: -len(kv[1])):
            L.append(f"- {shp}: {len(names)} x  {', '.join(names[:24])}{' ...' if len(names) > 24 else ''}")
    return "\n".join(L) + "\n"


def _scan_md(h) -> str:
    titles = {
        "nonfinite": "NaN / Inf", "constant": "Constant tensors", "tied": "Identical (tied or copied) tensors",
        "uniform_like": "Whole tensors that still look like their random init",
        "slices": "Slices that differ from the rest of their tensor (padding? never-trained channels? dead units?)",
        "io_dims": "Small interface dimensions (where padding usually hides)",
    }
    L = ["# Structural scan (auto-generated hints, NOT facts)\n",
         "Each hint is a question to answer from the code and a probe. Record the answers in ARCHITECTURE.md.",
         "Past win: an action head padded 6 -> 32 channels; only 6 were in the loss, the other 26 fed fresh noise back",
         "in at every denoising step. Zeroing those 26 input columns cut jitter 4-6x and lifted success 8/20 -> 17/20.\n"]
    for k, items in h.items():
        L.append(f"## {titles[k]} ({len(items)})\n")
        L += [f"- {x}" for x in items[:200]] or ["- none"]
        if len(items) > 200:
            L.append(f"- ... {len(items) - 200} more (see inventory.json)")
        L.append("")
    return "\n".join(L)


def _arch_md(stack, inv, groups) -> str:
    return f"""# Architecture: {stack.cfg.get('name') or stack.root.name}

<!-- AUTO-FACTS: this header block is regenerated until you edit below the line. Delete this comment once you
     start filling the file; from then on `editstack decode` never overwrites it. -->

Task: {stack.cfg.get('task') or '(fill in)'}
Weights: {inv['format']}, {inv['n_tensors']} tensors, {_fmt_params(inv['n_params'])} params, dtypes {inv['dtypes']}.
Inventory: INVENTORY.md. Structural hints: SCAN.md.

---------------------------------------------------------------------------------------------------------------

Fill every section from the CODE (model + training + inference), not from guesses. Mark each line
`[verified: <how>]` (code path:line, an offline probe, an equivalence test) or `[unverified]`.
Edits that rely on an [unverified] line are not allowed.

## 1. Inputs
Every input (images, text tokens, robot state, noise, time), its shape, preprocessing and normalisation stats
(where the mean/std live). Which channels are real data and which are padding.

## 2. Outputs and how they become actions/answers
Output tensor, which channels are used, de-normalisation, clipping. For generative heads: the sampling loop
(number of steps, step size, what is fed back in each step).

## 3. Training objective
Loss and its MASKS (e.g. `losses[:, :, :6]` means channels 6+ are never supervised). Which inputs were seen
in training and in which ranges (the in-distribution envelope).

## 4. Data flow
Block diagram: which stack feeds which, attention masks, where each modality enters, KV caching, which tensors
have no effect on the output (verify with an ablation probe).

## 5. Execution at deployment
How the outputs are executed (chunk length, re-planning, temporal ensembling, control rate). Differences between
sim and real (cameras, calibration, latency, lighting, shadows).

## 6. Exact identities you can rely on
Algebraic facts usable for exact edits (e.g. 'action space translation = bias shifts in in_proj/out_proj plus
time-feature term', 'pan output x k and pan input / k leaves holding unchanged'). Each with its equivalence test.
"""
