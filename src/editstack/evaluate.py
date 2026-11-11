"""Closed-loop evaluation of a variant on a fixed suite, with paired seeds and machine-wide RAM slots.

runs/<variant>/<suite>/s<seed>/
    meta.json      variant, chain_hash, harness_version, suite hash, episode list
    results.jsonl  one line per episode: {"episode", "seed", "metrics": {...}, "outcome", ...adapter extras}
    ep<i>/         per-episode artifacts written by the adapter (trajectories, videos)
    DONE           written when every planned episode of this seed has a result

A run directory whose chain_hash or harness_version no longer matches is STALE: it is moved to runs/_archive/
on the next eval (never silently mixed) and ignored by the leaderboard.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .stack import Stack, StackError


def suite_episodes(stack, suite: dict, adapter=None) -> List[dict]:
    if "episodes" in suite:
        return list(suite["episodes"])
    adapter = adapter or stack.adapter()
    if hasattr(adapter, "episodes"):
        return list(adapter.episodes(suite))
    raise StackError(f"suite {suite['name']} has no 'episodes' list and env/adapter.py has no episodes(suite)")


def suite_hash(eps: Sequence[dict]) -> str:
    return hashlib.sha256(json.dumps(list(eps), sort_keys=True).encode()).hexdigest()[:12]


def run_dir(stack, variant: str, suite: str, seed: int) -> Path:
    return stack.runs_dir / variant / suite / f"s{seed}"


def _parse_eps(spec: Optional[str], n: int) -> List[int]:
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    bad = [e for e in out if e < 0 or e >= n]
    if bad:
        raise StackError(f"episodes {bad} out of range 0..{n - 1}")
    return out


def done_episodes(d: Path) -> Dict[int, dict]:
    out = {}
    f = d / "results.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[int(r["episode"])] = r
    return out


def _archive(stack, d: Path, why: str) -> None:
    dest = stack.runs_dir / "_archive" / time.strftime("%Y%m%d-%H%M%S") / d.relative_to(stack.runs_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(dest))
    print(f"[eval] archived stale {d.relative_to(stack.root)} ({why}) -> {dest.relative_to(stack.root)}", flush=True)


# ------------------------------------------------------------------------------------------------ worker side

_W = {}


def _init_worker(root: str, variant: str, device: Optional[str]) -> None:
    os.environ["EDITSTACK_ROOT"] = root
    st = Stack(Path(root))
    from . import slots
    slots.acquire(st)
    adapter = st.adapter()
    applied = st.weights(variant)
    _W.update(stack=st, adapter=adapter, device=device,
              policy=adapter.load_policy(applied.tensors, st.adapter_ctx(device)))


def _run_one(task):
    seed, ep, spec, out_dir = task
    st, adapter = _W["stack"], _W["adapter"]
    ctx = st.adapter_ctx(_W["device"], Path(out_dir) / f"ep{ep}")
    t0 = time.time()
    try:
        r = adapter.run_episode(_W["policy"], dict(spec), int(seed), ctx)
    except Exception:
        return {"episode": ep, "seed": seed, "error": traceback.format_exc()[-2000:]}
    if not isinstance(r, dict) or "metrics" not in r:
        return {"episode": ep, "seed": seed, "error": "run_episode must return a dict with a 'metrics' dict"}
    r = {"episode": ep, "seed": seed, **r}
    r.setdefault("secs", round(time.time() - t0, 1))
    return r


# ------------------------------------------------------------------------------------------------ driver


def evaluate(stack, variant: str, suite_name: str, seeds: Sequence[int], episodes: Optional[str] = None,
             jobs: int = 1, device: Optional[str] = None, fresh: bool = False, note: str = "") -> List[Path]:
    from .variant import ORIGINAL, chain_hash
    if variant != ORIGINAL and not (stack.edits_dir / variant / "edit.py").exists():
        raise StackError(f"no variant {variant!r} (edits/{variant}/edit.py)")
    suite = stack.suite(suite_name)
    eps = suite_episodes(stack, suite)
    if not eps:
        raise StackError(f"suite {suite_name} has no episodes - define them in eval/suites/{suite_name}.json "
                         f"(AGENTS.md Phase 0)")
    want = _parse_eps(episodes, len(eps))
    ch = chain_hash(stack, variant) if variant != ORIGINAL else "original"
    sh = suite_hash(eps)
    tasks, dirs = [], []
    for seed in seeds:
        d = run_dir(stack, variant, suite_name, seed)
        if (d / "meta.json").exists():
            m = json.loads((d / "meta.json").read_text())
            why = []
            if m.get("chain_hash") != ch:
                why.append(f"edit changed {m.get('chain_hash')} -> {ch}")
            if m.get("harness_version") != stack.harness_version:
                why.append(f"harness {m.get('harness_version')} -> {stack.harness_version}")
            if m.get("suite_hash") != sh:
                why.append("suite episodes changed")
            if why or fresh:
                _archive(stack, d, "; ".join(why) or "--fresh")
        d.mkdir(parents=True, exist_ok=True)
        have = done_episodes(d)
        meta = {"variant": variant, "chain_hash": ch, "harness_version": stack.harness_version, "suite": suite_name,
                "suite_hash": sh, "suite_role": suite.get("role", "selection"), "seed": seed,
                "episodes_planned": sorted(set(want) | set(have)), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pid": os.getpid(), "note": note}
        (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        (d / "DONE").unlink(missing_ok=True)
        tasks += [(seed, e, eps[e], str(d)) for e in want if e not in have]
        dirs.append(d)
    if not tasks:
        print("[eval] nothing to do: all requested episodes already have results", flush=True)
        for d in dirs:
            _write_done(stack, d)
        return dirs
    jobs = max(1, min(jobs, len(tasks)))
    print(f"[eval] {variant} on {suite_name}: {len(tasks)} episodes, seeds {list(seeds)}, {jobs} process(es)", flush=True)
    t0 = time.time()
    errors = 0
    sink = {d: open(d / "results.jsonl", "a") for d in dirs}
    by_seed = {int(Path(d).name[1:]): d for d in dirs}
    try:
        if jobs == 1:
            _init_worker(str(stack.root), variant, device)
            results = map(_run_one, tasks)
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(jobs, initializer=_init_worker, initargs=(str(stack.root), variant, device))
            results = pool.imap_unordered(_run_one, tasks)
        for k, r in enumerate(results, 1):
            if "error" in r:
                errors += 1
                print(f"[eval] episode {r['episode']} seed {r['seed']} FAILED:\n{r['error']}", file=sys.stderr, flush=True)
                if errors >= 3 and errors == k:
                    raise StackError("first 3 episodes all failed - fix env/adapter.py before burning more compute")
                continue
            f = sink[by_seed[int(r["seed"])]]
            f.write(json.dumps(r) + "\n")
            f.flush()
            m = " ".join(f"{a}={int(bool(v)) if isinstance(v, bool) else v}" for a, v in r["metrics"].items())
            print(f"[eval] {k}/{len(tasks)} s{r['seed']} ep{r['episode']} {r.get('outcome', '')} {m} "
                  f"({r.get('secs', '?')}s)", flush=True)
        if jobs > 1:
            pool.close()
            pool.join()
    finally:
        for f in sink.values():
            f.close()
    for d in dirs:
        _write_done(stack, d)
    print(f"[eval] finished in {time.time() - t0:.0f}s, {errors} failed episodes", flush=True)
    return dirs


def _write_done(stack, d: Path) -> None:
    meta = json.loads((d / "meta.json").read_text())
    have = done_episodes(d)
    if all(e in have for e in meta["episodes_planned"]):
        metrics = stack.cfg["metrics"]["all"]
        summ = {m: sum(bool(r["metrics"].get(m)) for r in have.values()) for m in metrics}
        (d / "DONE").write_text(json.dumps({"n": len(have), **summ, "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")


def detach(stack, argv: List[str]) -> Path:
    """Re-run this CLI command in a new session (survives the agent's shell / background-job time limits)."""
    log_dir = stack.runs_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"
    cmd = [sys.executable, "-m", "editstack", *[a for a in argv if a != "--detach"]]
    env = dict(os.environ, EDITSTACK_ROOT=str(stack.root), PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        import subprocess
        subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd=stack.root)
    return log
