"""Machine-wide RAM admission for model processes: ONE budget for the whole machine, shared by every stack.

Lesson: parallel agents once drove requested RAM to ~110 GB because each believed it used little (ps RSS showed
1-4 GB; the real macOS `footprint` of a VLA process was ~8 GB). Honour-system caps failed; a lock did not.

Machine config (optional, recommended): ~/.config/editstack/machine.json
    {"ram_budget_gb": 60,
     "lock_dir": "~/.cache/editstack/slots",
     "legacy_slot_dirs": ["/tmp/smolvla_slots"],   # other lock systems on this machine (old slots.py format)
     "legacy_max_procs": 6, "legacy_est_gb": 9}
The machine budget wins over any stack's ram_budget_gb (the smaller of the two is used). Stacks only say how much
one of THEIR processes needs (resources.per_process_gb, measured with `footprint`, not ps RSS).

Each model process calls acquire(stack): under an admission lock it sums live reservations (ours + legacy
holders) and is admitted only if sum + per_process_gb <= budget. A reservation is a file held under flock; the OS
drops the lock when the process dies, so crashes never leak RAM budget. With legacy dirs configured, the process
also holds one legacy slot file, so the legacy system's process cap counts editstack processes too.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

_held = None          # (fd, path, legacy_fd, legacy_path)
MACHINE = Path(os.path.expanduser("~/.config/editstack/machine.json"))


def machine_cfg() -> dict:
    try:
        return json.loads(MACHINE.read_text())
    except (OSError, ValueError):
        return {}


def _dir() -> Path:
    p = Path(os.path.expanduser(machine_cfg().get("lock_dir", "~/.cache/editstack/slots")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def budget_gb(stack=None) -> float:
    m = machine_cfg().get("ram_budget_gb")
    s = stack.cfg["resources"].get("ram_budget_gb") if stack else None
    vals = [float(v) for v in (m, s) if v is not None]
    return min(vals) if vals else 60.0


def _locked(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return True
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    return False


def reservations(cleanup: bool = False):
    """Live reservations. Files of dead holders are removed only when cleanup=True (caller holds admit.lock),
    so a process that has just created its file but not yet locked it is never deleted from under it."""
    out = []
    for p in sorted(_dir().glob("res_*")):
        mine = _held is not None and p == _held[1]
        if not mine and not _locked(p):
            if cleanup:
                try:
                    p.unlink()
                except OSError:
                    pass
            continue
        try:
            pid, gb, root, *cmd = p.read_text().split("\t")
            out.append({"pid": int(pid), "gb": float(gb), "stack": root, "cmd": "\t".join(cmd).strip(), "kind": "editstack"})
        except (OSError, ValueError):
            out.append({"pid": -1, "gb": 0.0, "stack": "?", "cmd": "(initialising)", "kind": "editstack"})
    m = machine_cfg()
    for d in m.get("legacy_slot_dirs", []):
        for i in range(int(m.get("legacy_max_procs", 6))):
            p = Path(d) / f"slot_{i}"
            if p.exists() and _locked(p):
                if _held is not None and _held[3] == p:
                    continue          # our own legacy slot: already counted via our reservation
                try:
                    txt = p.read_text().split()
                    pid = int(txt[0])
                    gb = int(txt[1]) / 1000 if len(txt) > 1 and txt[1].isdigit() else float(m.get("legacy_est_gb", 9))
                except (OSError, ValueError, IndexError):
                    pid, gb = -1, float(m.get("legacy_est_gb", 9))
                out.append({"pid": pid, "gb": gb, "stack": d, "cmd": "legacy slot", "kind": "legacy"})
    return out


def _take_legacy_slot(gb: float):
    m = machine_cfg()
    for d in m.get("legacy_slot_dirs", []):
        Path(d).mkdir(parents=True, exist_ok=True)
        for i in range(int(m.get("legacy_max_procs", 6))):
            p = Path(d) / f"slot_{i}"
            fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {int(gb * 1000)} editstack {' '.join(sys.argv)[:200]}\n".encode())
            return fd, p
        return None          # legacy process cap full
    return (None, None)      # no legacy system configured


def acquire(stack, gb: float = None, poll: float = 5.0) -> None:
    """Block until a reservation of `gb` (default: stack per_process_gb) fits the machine budget. Idempotent."""
    global _held
    if _held is not None:
        return
    gb = float(gb if gb is not None else stack.cfg["resources"]["per_process_gb"])
    budget = budget_gb(stack)
    if gb > budget:
        raise RuntimeError(f"one process needs {gb} GB > machine budget {budget} GB")
    d = _dir()
    legacy_admits = [Path(x) / "admit.lock" for x in machine_cfg().get("legacy_slot_dirs", [])]
    waited, t0 = False, time.time()
    while True:
        fds = []
        try:
            for lock in [d / "admit.lock", *legacy_admits]:
                lock.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o666)
                fcntl.flock(fd, fcntl.LOCK_EX)
                fds.append(fd)
            used = sum(r["gb"] for r in reservations(cleanup=True))
            if used + gb <= budget:
                legacy = _take_legacy_slot(gb)
                if legacy is not None:
                    p = d / f"res_{os.getpid()}"
                    fd = os.open(p, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o666)
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    os.write(fd, f"{os.getpid()}\t{gb}\t{stack.root}\t{' '.join(sys.argv)[:300]}".encode())
                    _held = (fd, p, legacy[0], legacy[1])
                    import atexit
                    atexit.register(release)
                    if waited:
                        print(f"[slots] admitted after {time.time() - t0:.0f}s ({used:.0f}+{gb:.0f} of {budget:.0f} GB)",
                              file=sys.stderr, flush=True)
                    return
        finally:
            for fd in fds:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        if not waited:
            print(f"[slots] waiting: {used:.0f} GB reserved + {gb:.0f} GB > machine budget {budget:.0f} GB "
                  f"(or legacy process cap full)", file=sys.stderr, flush=True)
            waited = True
        time.sleep(poll)


def release() -> None:
    global _held
    if _held is None:
        return
    fd, p, lfd, lp = _held
    for f, path in ((fd, p), (lfd, lp)):
        if f is None:
            continue
        if path == p:
            try:
                path.unlink()
            except OSError:
                pass
        else:
            os.ftruncate(f, 0)
        fcntl.flock(f, fcntl.LOCK_UN)
        os.close(f)
    _held = None


def status(stack) -> str:
    rs = reservations()
    per = stack.cfg["resources"]["per_process_gb"]
    budget = budget_gb(stack)
    lines = [f"{r['pid']:>7d} {r['gb']:5.1f} GB  {r['kind']:9s} {Path(r['stack']).name:20s} {r['cmd'][:80]}" for r in rs]
    used = sum(r["gb"] for r in rs)
    src = str(MACHINE) if MACHINE.exists() else "no machine.json: using the stack's ram_budget_gb"
    lines.append(f"reserved {used:.0f} of {budget:.0f} GB machine-wide ({src}); a new process of this stack needs "
                 f"{per} GB -> {max(0, int((budget - used) // per))} more fit now")
    return "\n".join(lines)
