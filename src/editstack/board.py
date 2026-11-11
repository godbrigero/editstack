"""Leaderboard, qualification, paired comparisons and promotion to weights/best.

Rules (stack.json "qualify") encode the lessons:
  - min seeds per suite: two seeds misled twice in past runs (23/24 -> 31/36 and 24/24 -> 31/36 at the third seed)
  - an in-distribution floor so a variant cannot win by breaking the cases the model already handled
  - ranking only on suites with role "selection"; a "heldout" suite (never used for picking) must be run before
    promotion, because dozens of variants selected on the same placements overfit them (winner's curse)
Runs from an older harness_version or a changed edit chain are ignored.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .evaluate import done_episodes


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def collect(stack) -> Dict[str, Dict[str, dict]]:
    """variant -> suite -> {"rows": {(seed, ep): result}, "seeds": set, "stale": n_dirs_ignored}"""
    from .variant import ORIGINAL, chain_hash, EditError
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    current = {}
    if not stack.runs_dir.exists():
        return out
    from .evaluate import suite_episodes, suite_hash
    shash = {}
    for name, s in stack.suites().items():
        try:
            shash[name] = suite_hash(suite_episodes(stack, s))
        except Exception:
            shash[name] = None
    for meta_f in sorted(stack.runs_dir.glob("*/*/s*/meta.json")):
        d = meta_f.parent
        variant, suite = d.parent.parent.name, d.parent.name
        if variant.startswith("_"):
            continue
        m = json.loads(meta_f.read_text())
        if variant not in current:
            try:
                current[variant] = "original" if variant == ORIGINAL else chain_hash(stack, variant)
            except (EditError, OSError):
                current[variant] = None
        slot = out[variant].setdefault(suite, {"rows": {}, "seeds": set(), "stale": 0,
                                                "role": m.get("suite_role", "selection")})
        if (m.get("harness_version") != stack.harness_version or m.get("chain_hash") != current[variant]
                or m.get("suite_hash") != shash.get(suite)):
            slot["stale"] += 1
            continue
        rows = done_episodes(d)
        if rows:
            slot["seeds"].add(int(m["seed"]))
            for ep, r in rows.items():
                slot["rows"][(int(m["seed"]), ep)] = r
    return out


def summarize(stack, data=None) -> List[dict]:
    from .variant import ORIGINAL, load_edit, EditError
    data = data if data is not None else collect(stack)
    cfg = stack.cfg
    metrics = cfg["metrics"]["all"]
    primary = cfg["metrics"]["primary"]
    q = cfg["qualify"]
    roles = {n: s.get("role", "selection") for n, s in stack.suites().items()}
    board = []
    for variant, suites in data.items():
        row = {"variant": variant, "suites": {}, "qualified": True, "why": []}
        for sname, s in suites.items():
            rs = list(s["rows"].values())
            n = len(rs)
            row["suites"][sname] = {"n": n, "seeds": len(s["seeds"]), "stale": s["stale"], "role": roles.get(sname, s["role"]),
                                    **{m: sum(bool(r["metrics"].get(m)) for r in rs) for m in metrics}}
        for rule in q["rules"]:
            s = row["suites"].get(rule["suite"], {"n": 0, "seeds": 0})
            if s["seeds"] < rule.get("min_seeds", 0):
                row["qualified"] = False
                row["why"].append(f"{rule['suite']}: {s['seeds']}/{rule.get('min_seeds')} seeds")
            if s["n"] < rule.get("min_episodes", 0):
                row["qualified"] = False
                row["why"].append(f"{rule['suite']}: {s['n']}/{rule.get('min_episodes')} episodes")
            for m, r in rule.get("min_rate", {}).items():
                if s["n"] and s.get(m, 0) / s["n"] < r:
                    row["qualified"] = False
                    row["why"].append(f"{rule['suite']} {m} {s.get(m, 0)}/{s['n']} < {r:.0%}")
        key = []
        for sname, m in q["rank"]:
            s = row["suites"].get(sname)
            key.append(s[m] / s["n"] if s and s["n"] else -1)
        row["key"] = key
        # paired comparison against the parent on every shared (suite, seed, episode)
        parent = None
        if variant != ORIGINAL:
            try:
                parent = getattr(load_edit(stack, variant), "PARENT", None) or ORIGINAL
            except (EditError, OSError):
                parent = None
        row["parent"] = parent
        row["vs_parent"] = {}
        if parent and parent in data:
            for sname, s in suites.items():
                ps = data[parent].get(sname)
                if not ps:
                    continue
                shared = set(s["rows"]) & set(ps["rows"])
                w = sum(bool(s["rows"][k]["metrics"].get(primary)) and not bool(ps["rows"][k]["metrics"].get(primary)) for k in shared)
                l = sum(not bool(s["rows"][k]["metrics"].get(primary)) and bool(ps["rows"][k]["metrics"].get(primary)) for k in shared)
                if shared:
                    row["vs_parent"][sname] = {"pairs": len(shared), "wins": w, "losses": l, "p": sign_test(w, l)}
        hold = [sn for sn, s in row["suites"].items() if s["role"] == "heldout" and s["n"]]
        row["heldout_done"] = bool(hold)
        board.append(row)
    board.sort(key=lambda r: (r["qualified"], r["key"]), reverse=True)
    return board


def render(stack, board: List[dict]) -> str:
    metrics = stack.cfg["metrics"]["all"]
    suites = sorted({s for r in board for s in r["suites"]}, key=lambda s: (stack.suites().get(s, {}).get("role", "z"), s))
    L = [f"# Leaderboard ({time.strftime('%Y-%m-%d %H:%M')}, harness {stack.harness_version})\n",
         f"Qualify: {json.dumps(stack.cfg['qualify']['rules'])}",
         f"Rank by: {stack.cfg['qualify']['rank']}. Counts are successes/episodes; [lo-hi] = 95% Wilson interval.",
         f"vs parent = paired wins/losses on `{stack.cfg['metrics']['primary']}` over identical (seed, episode); p = sign test.\n"]
    head = "| variant | qual | " + " | ".join(f"{s} ({stack.suites().get(s, {}).get('role', '?')})" for s in suites) + " | vs parent |"
    L += [head, "|" + "---|" * (len(suites) + 3)]
    for r in board:
        cells = []
        for s in suites:
            x = r["suites"].get(s)
            if not x or not x["n"]:
                cells.append("-" + (f" ({x['stale']} stale)" if x and x["stale"] else ""))
                continue
            parts = []
            for m in metrics:
                lo, hi = wilson(x[m], x["n"])
                parts.append(f"{m} {x[m]}/{x['n']} [{lo:.0%}-{hi:.0%}]")
            cells.append("<br>".join(parts) + f"<br>{x['seeds']} seeds")
        vp = "; ".join(f"{s}: +{v['wins']}/-{v['losses']} p={v['p'] if v['p'] >= 0.001 else '<0.001'}"
                       for s, v in r["vs_parent"].items())
        q = "yes" if r["qualified"] else "no: " + ", ".join(r["why"][:3])
        L.append(f"| **{r['variant']}** ({r['parent'] or '-'}) | {q} | " + " | ".join(cells) + f" | {vp or '-'} |")
    best = stack.best_link
    L.append(f"\nweights/best -> {os.readlink(best) if best.is_symlink() else '(none)'}")
    return "\n".join(L) + "\n"


def write(stack) -> str:
    board = summarize(stack)
    md = render(stack, board)
    stack.board_dir.mkdir(parents=True, exist_ok=True)
    (stack.board_dir / "LEADERBOARD.md").write_text(md)
    return md


def promote(stack, dry_run: bool = False, force: bool = False, variant: Optional[str] = None) -> str:
    from .variant import export, ORIGINAL
    board = summarize(stack)
    q = [r for r in board if r["qualified"]]
    if variant:
        q = [r for r in q if r["variant"] == variant]
    if not q:
        return "no qualified variant" + (f" named {variant}" if variant else "") + "; weights/best unchanged"
    win = q[0]
    name = win["variant"]
    msgs = []
    if stack.cfg["qualify"].get("require_heldout") and not win["heldout_done"]:
        held = [n for n, s in stack.suites().items() if s.get("role") == "heldout"]
        if held and not force:
            return (f"{name} leads but has no held-out results. Run: editstack eval {name} --suite {held[0]} "
                    f"--seeds 0 ; then promote again (or --force).")
    for sname, s in win["suites"].items():
        if s["role"] == "heldout" and s["n"]:
            sel = [x for x in win["suites"].values() if x["role"] == "selection" and x["n"]]
            p = stack.cfg["metrics"]["primary"]
            if sel:
                sel_rate = sum(x[p] for x in sel) / sum(x["n"] for x in sel)
                gap = sel_rate - s[p] / s["n"]
                if gap > 0.15:
                    msgs.append(f"WARNING: {name} held-out {p} is {gap:.0%} below its selection suites (overfit?)")
    current = os.readlink(stack.best_link) if stack.best_link.is_symlink() else None
    target = stack.variants_dir / name
    if dry_run:
        return "\n".join(msgs + [f"would promote {name} (current: {current})"])
    if name == ORIGINAL:
        target = stack.original_dir
    else:
        man = target / "MANIFEST.json"
        from .variant import chain_hash
        if not man.exists() or json.loads(man.read_text()).get("chain_hash") != chain_hash(stack, name):
            msgs.append(f"exporting {name} ...")
            export(stack, name)
    rel = os.path.relpath(target, stack.best_link.parent)
    if current == rel:
        return "\n".join(msgs + [f"weights/best already -> {name}"])
    if stack.best_link.is_symlink() or stack.best_link.exists():
        stack.best_link.unlink()
    stack.best_link.symlink_to(rel)
    hist = stack.board_dir / "BEST_HISTORY.md"
    hist.parent.mkdir(parents=True, exist_ok=True)
    cells = ", ".join(f"{s} " + "/".join(f"{m} {x[m]}/{x['n']}" for m in stack.cfg["metrics"]["all"])
                      for s, x in win["suites"].items() if x["n"])
    with open(hist, "a") as f:
        if f.tell() == 0:
            f.write("# weights/best history\n\n")
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')}: **{name}** (harness {stack.harness_version}) {cells} (was: {current})\n")
    write(stack)
    return "\n".join(msgs + [f"weights/best -> {name}"])
