"""Closed-loop evaluation of a variant on a fixed suite, with paired seeds and machine-wide RAM slots.

runs/<variant>/<suite>/s<seed>/
    meta.json      variant, chain_hash, harness_version, metric_version, adapter hash, suite hash, episode list
    results.jsonl  one line per episode: {"episode", "seed", "metrics": {...}, "outcome", ...adapter extras}
    ep<i>/         per-episode artifacts written by the adapter (trajectories, observations)
    DONE           written when every planned episode of this seed has a result
    .lock          held (flock) by the process running this seed; a second eval of the same dir refuses to start

A run directory whose chain_hash, harness_version, adapter or suite no longer matches is STALE: it is moved to
runs/_archive/ on the next eval (never silently mixed) and ignored by the leaderboard. If only metric_version
changed, `editstack rescore` can update results from stored artifacts without re-simulating.
"""
from __future__ import annotations

import fcntl
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


PROBE_ONLY = {"capture", "load_obs", "rescore"}     # adding/changing these never changes episode results


def adapter_hash(stack) -> str:
    """Fingerprint of the parts of env/adapter.py that can change episode results. Probe-only helpers (capture,
    load_obs, rescore), comments and docstrings are excluded, so adding a probe helper does not invalidate runs."""
    import ast
    p = stack.root / "env" / "adapter.py"
    if not p.exists():
        return "none"
    src = p.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return hashlib.sha256(src.encode()).hexdigest()[:12]
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in PROBE_ONLY:
            continue
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            continue    # module docstring
        for sub in ast.walk(node):    # strip docstrings inside functions/classes
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and sub.body and \
                    isinstance(sub.body[0], ast.Expr) and isinstance(getattr(sub.body[0], "value", None), ast.Constant):
                sub.body = sub.body[1:] or [ast.Pass()]
        keep.append(ast.dump(node, annotate_fields=False))
    return hashlib.sha256("\n".join(keep).encode()).hexdigest()[:12]


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


def done_episodes(d: Path, warn: bool = False) -> Dict[int, dict]:
    """Results by episode. Corrupt/truncated lines (e.g. a crash mid-write) are skipped, so the episode reruns."""
    out = {}
    f = d / "results.jsonl"
    if f.exists():
        for k, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                out[int(r["episode"])] = r
            except (ValueError, KeyError, TypeError):
                if warn:
                    print(f"[eval] {f}: skipping corrupt line {k + 1}", file=sys.stderr)
    return out


def _archive(stack, d: Path, why: str) -> None:
    dest = stack.runs_dir / "_archive" / time.strftime("%Y%m%d-%H%M%S") / d.relative_to(stack.runs_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(dest))
    print(f"[eval] archived stale {d.relative_to(stack.root)} ({why}) -> {dest.relative_to(stack.root)}", flush=True)


def lock_dir(d: Path):
    """Exclusive, non-blocking lock on a run dir. Returns the fd or raises StackError naming the holder."""
    d.mkdir(parents=True, exist_ok=True)
    fd = os.open(d / ".lock", os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = (d / ".lock").read_text().strip() or "?"
        os.close(fd)
        raise StackError(f"{d} is being evaluated by pid {holder}. Use `editstack wait` instead of starting it again.")
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    return fd


def is_locked(d: Path) -> bool:
    p = d / ".lock"
    if not p.exists():
        return False
    fd = os.open(p, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        os.close(fd)
    return False


# ------------------------------------------------------------------------------------------------ worker side

_W = {}


def _init_worker(root: str, variant: str, device: Optional[str]) -> None:
    """Never raises: a failing initializer would make multiprocessing respawn workers forever (each reloading the
    model). The error is stored and returned by every task instead, and the driver aborts."""
    try:
        os.environ["EDITSTACK_ROOT"] = root
        st = Stack(Path(root))
        from . import slots
        slots.acquire(st)
        adapter = st.adapter()
        applied = st.weights(variant)
        _W.update(stack=st, adapter=adapter, device=device,
                  policy=adapter.load_policy(applied.tensors, st.adapter_ctx(device)))
    except BaseException:
        _W["init_error"] = traceback.format_exc()[-3000:]


def _run_one(task):
    seed, ep, spec, out_dir = task
    if "init_error" in _W:
        return {"episode": ep, "seed": seed, "error": "worker init failed:\n" + _W["init_error"], "fatal": True}
    st, adapter = _W["stack"], _W["adapter"]
    ctx = st.adapter_ctx(_W["device"], Path(out_dir) / f"ep{ep}")
    t0 = time.time()
    try:
        r = adapter.run_episode(_W["policy"], dict(spec), int(seed), ctx)
    except Exception:
        return {"episode": ep, "seed": seed, "error": traceback.format_exc()[-2000:]}
    if not isinstance(r, dict) or not isinstance(r.get("metrics"), dict):
        return {"episode": ep, "seed": seed, "error": "run_episode must return a dict with a 'metrics' dict", "fatal": True}
    missing = [m for m in st.cfg["metrics"]["all"] if m not in r["metrics"]]
    if missing:
        return {"episode": ep, "seed": seed, "error": f"run_episode metrics lack {missing} (stack.json metrics.all)", "fatal": True}
    r = {"episode": ep, "seed": seed, **r}
    r.setdefault("secs", round(time.time() - t0, 1))
    return r


# ------------------------------------------------------------------------------------------------ driver


def evaluate(stack, variant: str, suite_name: str, seeds: Sequence[int], episodes: Optional[str] = None,
             jobs: int = 1, device: Optional[str] = None, fresh: bool = False, note: str = "") -> List[Path]:
    from .variant import ORIGINAL, chain_hash, apply_variant, variant_exists
    from .decode import quick_verify
    if not variant_exists(stack, variant):
        raise StackError(f"no variant {variant!r} (edits/{variant}/edit.py)")
    quick_verify(stack)
    suite = stack.suite(suite_name)
    eps = suite_episodes(stack, suite)
    if not eps:
        raise StackError(f"suite {suite_name} has no episodes - define them in eval/suites/{suite_name}.json "
                         f"(AGENTS.md Phase 0)")
    want = _parse_eps(episodes, len(eps))
    ch = chain_hash(stack, variant) if variant != ORIGINAL else "original"
    sh, ah = suite_hash(eps), adapter_hash(stack)
    mv = str(stack.cfg.get("metric_version", "1"))
    # apply once in the driver: edit errors and claim conflicts fail here, before any worker loads a model
    if variant != ORIGINAL:
        a = apply_variant(stack, variant, register=True)
        print(f"[eval] {variant}: {len(a.changed)} tensors changed vs original (chain {' -> '.join(a.chain)})", flush=True)
        del a
    tasks, dirs, locks = [], [], []
    try:
        for seed in seeds:
            d = run_dir(stack, variant, suite_name, seed)
            if d.exists() and is_locked(d):
                raise StackError(f"{d.relative_to(stack.root)} is already being evaluated. Use `editstack wait {variant} "
                                 f"--suite {suite_name}`; don't start it twice.")
            if (d / "meta.json").exists():
                m = json.loads((d / "meta.json").read_text())
                why = []
                if m.get("chain_hash") != ch:
                    why.append(f"weights changed {m.get('chain_hash')} -> {ch}")
                if m.get("harness_version") != stack.harness_version:
                    why.append(f"harness {m.get('harness_version')} -> {stack.harness_version}")
                if m.get("adapter_hash") != ah:
                    why.append("env/adapter.py changed")
                if m.get("suite_hash") != sh:
                    why.append("suite episodes changed")
                if m.get("metric_version", "1") != mv and not why:
                    raise StackError(f"{d.relative_to(stack.root)} was scored with metric_version {m.get('metric_version', '1')}; "
                                     f"run `editstack rescore` (no sim needed) or pass --fresh")
                if why or fresh:
                    _archive(stack, d, "; ".join(why) or "--fresh")
            locks.append(lock_dir(d))
            have = done_episodes(d, warn=True)
            meta = {"variant": variant, "chain_hash": ch, "harness_version": stack.harness_version, "metric_version": mv,
                    "adapter_hash": ah, "suite": suite_name, "suite_hash": sh, "suite_size": len(eps),
                    "suite_role": suite.get("role", "selection"), "seed": seed,
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
        _run(stack, variant, suite_name, seeds, tasks, dirs, jobs, device)
        return dirs
    finally:
        for fd in locks:
            os.close(fd)


def _run(stack, variant, suite_name, seeds, tasks, dirs, jobs, device) -> None:
    jobs = max(1, min(jobs, len(tasks)))
    print(f"[eval] {variant} on {suite_name}: {len(tasks)} episodes, seeds {list(seeds)}, {jobs} process(es)", flush=True)
    t0 = time.time()
    errors = done = 0
    sink = {d: open(d / "results.jsonl", "a") for d in dirs}
    by_seed = {int(Path(d).name[1:]): d for d in dirs}
    pool = None
    try:
        if jobs == 1:
            _init_worker(str(stack.root), variant, device)
            results = map(_run_one, tasks)
        else:
            pool = mp.get_context("spawn").Pool(jobs, initializer=_init_worker, initargs=(str(stack.root), variant, device))
            results = pool.imap_unordered(_run_one, tasks)
        for k, r in enumerate(results, 1):
            if "error" in r:
                errors += 1
                print(f"[eval] episode {r['episode']} seed {r['seed']} FAILED:\n{r['error']}", file=sys.stderr, flush=True)
                if r.get("fatal") or (errors >= 3 and done == 0):
                    raise StackError("aborting: " + ("fatal adapter/worker error" if r.get("fatal") else
                                                     "first 3 episodes all failed") + " - fix it before burning more compute")
                continue
            done += 1
            f = sink[by_seed[int(r["seed"])]]
            f.write(json.dumps(r) + "\n")
            f.flush()
            m = " ".join(f"{a}={int(bool(v)) if isinstance(v, bool) else v}" for a, v in r["metrics"].items())
            print(f"[eval] {k}/{len(tasks)} s{r['seed']} ep{r['episode']} {r.get('outcome', '')} {m} "
                  f"({r.get('secs', '?')}s)", flush=True)
        if pool:
            pool.close()
            pool.join()
    except BaseException:
        if pool:
            pool.terminate()
        raise
    finally:
        for f in sink.values():
            f.close()
        for d in dirs:
            _write_done(stack, d)
    print(f"[eval] finished in {time.time() - t0:.0f}s, {errors} failed episodes", flush=True)


def _write_done(stack, d: Path) -> None:
    meta = json.loads((d / "meta.json").read_text())
    have = done_episodes(d)
    if all(e in have for e in meta["episodes_planned"]):
        metrics = stack.cfg["metrics"]["all"]
        summ = {m: sum(bool(r["metrics"].get(m)) for r in have.values()) for m in metrics}
        full = len(have) >= meta.get("suite_size", 0)
        (d / "DONE").write_text(json.dumps({"n": len(have), "full_suite": full, **summ,
                                            "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")


# ------------------------------------------------------------------------------------------------ jobs (detach / wait)


def jobs_dir(stack) -> Path:
    d = stack.runs_dir / "_jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def detach(stack, argv: List[str], variant: str, suite: str, seeds: Sequence[int]) -> dict:
    """Re-run this CLI command in a new session (survives the agent's shell / background-job time limits).
    Writes runs/_jobs/<id>.json BEFORE returning, so `editstack wait` can never race the child."""
    import subprocess
    jid = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    log = stack.runs_dir / "_logs" / f"{jid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "editstack", *[a for a in argv if a != "--detach"]]
    env = dict(os.environ, EDITSTACK_ROOT=str(stack.root), PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd=stack.root)
    job = {"id": jid, "pid": p.pid, "variant": variant, "suite": suite, "seeds": list(seeds), "log": str(log),
           "started": time.time()}
    (jobs_dir(stack) / f"{jid}.json").write_text(json.dumps(job, indent=1) + "\n")
    return job


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:   # reap our own zombie children
        os.waitpid(pid, os.WNOHANG)
    except OSError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def latest_job(stack, variant: str, suite: Optional[str]) -> Optional[dict]:
    jobs = []
    for f in jobs_dir(stack).glob("*.json"):
        j = json.loads(f.read_text())
        if j["variant"] == variant and (suite is None or j["suite"] == suite):
            jobs.append(j)
    return max(jobs, key=lambda j: j["started"]) if jobs else None


def wait(stack, variant: str, suite: Optional[str], seeds: Optional[Sequence[int]], timeout: float) -> List[str]:
    """Block until the latest detached job for (variant, suite) exits; then report its seeds.
    Without a job, wait on live run-dir locks. Stale DONE markers from older runs are never trusted."""
    t0 = time.time()
    job = latest_job(stack, variant, suite)
    if job:
        while pid_alive(job["pid"]):
            if time.time() - t0 > timeout:
                raise StackError(f"timeout after {timeout:.0f}s; job {job['id']} still running (log {job['log']})")
            time.sleep(5)
        suite, seeds = job["suite"], seeds or job["seeds"]
    else:
        pat = f"{variant}/{suite or '*'}/s*"
        while any(is_locked(d) for d in stack.runs_dir.glob(pat)):
            if time.time() - t0 > timeout:
                raise StackError(f"timeout after {timeout:.0f}s")
            time.sleep(5)
    out = []
    suites = [suite] if suite else sorted(p.name for p in (stack.runs_dir / variant).iterdir() if p.is_dir())
    for s in suites:
        for d in sorted((stack.runs_dir / variant / s).glob("s*")):
            if seeds is not None and int(d.name[1:]) not in seeds:
                continue
            done = d / "DONE"
            if done.exists() and (not job or done.stat().st_mtime >= job["started"] - 1):
                out.append(f"{d.relative_to(stack.root)}: {done.read_text().strip()}")
            else:
                out.append(f"{d.relative_to(stack.root)}: NOT COMPLETE" + (f" - see {job['log']}" if job else ""))
    return out


# ------------------------------------------------------------------------------------------------ rescore


def rescore(stack, variant: Optional[str] = None) -> List[str]:
    """Recompute metrics from stored artifacts with adapter.rescore(result, ep_dir, ctx) -> result, for runs whose
    metric_version differs from stack.json (no sim). Keeps a backup of every rewritten results.jsonl."""
    adapter = stack.adapter()
    if not hasattr(adapter, "rescore"):
        raise StackError("env/adapter.py has no rescore(result, ep_dir, ctx); old runs must be re-simulated (--fresh)")
    mv = str(stack.cfg.get("metric_version", "1"))
    out = []
    for meta_f in sorted(stack.runs_dir.glob(f"{variant or '*'}/*/s*/meta.json")):
        d = meta_f.parent
        if d.parts[-3].startswith("_"):
            continue
        m = json.loads(meta_f.read_text())
        if m.get("metric_version", "1") == mv:
            continue
        if is_locked(d):
            out.append(f"skip {d.relative_to(stack.root)} (running)")
            continue
        rows = done_episodes(d)
        new = [adapter.rescore(dict(r), d / f"ep{ep}", stack.adapter_ctx(None, d / f"ep{ep}")) for ep, r in sorted(rows.items())]
        shutil.copy2(d / "results.jsonl", d / f"results.metric_v{m.get('metric_version', '1')}.jsonl")
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in new))
        m["metric_version"] = mv
        meta_f.write_text(json.dumps(m, indent=1) + "\n")
        if (d / "DONE").exists():
            _write_done(stack, d)
        out.append(f"rescored {d.relative_to(stack.root)} ({len(new)} episodes)")
    return out or ["nothing to rescore"]
