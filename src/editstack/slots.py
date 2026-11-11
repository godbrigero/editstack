"""Machine-wide RAM admission for model processes, shared by every stack on the machine.

Lesson: parallel agents once drove requested RAM to ~110 GB because each believed it used little (ps RSS showed
1-4 GB; the real macOS `footprint` of a VLA process was ~8 GB). Honour-system caps failed; a lock did not.

Each process that loads a model calls acquire(stack): it reserves `per_process_gb` (from stack.json) and is admitted
only if the sum of live reservations + its own <= ram_budget_gb. Reservations are files held under an exclusive
flock; the OS drops the lock when the process dies, so crashed processes never leak a reservation.
Lock dir default: ~/.cache/editstack/slots (shared by all stacks so the budget is truly machine-wide).
"""
from __future__ import annotations

import fcntl
import os
import sys
import time
from pathlib import Path

_held = None


def _dir(stack=None, lock_dir=None) -> Path:
    d = lock_dir or (stack.cfg["resources"]["lock_dir"] if stack else "~/.cache/editstack/slots")
    p = Path(os.path.expanduser(d))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _alive(path: Path) -> bool:
    """A reservation is live iff someone holds its flock."""
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


def reservations(stack=None, lock_dir=None):
    d = _dir(stack, lock_dir)
    out = []
    for p in sorted(d.glob("res_*")):
        if _held is not None and p == _held[1]:
            alive = True
        else:
            alive = _alive(p)
        if not alive:
            try:
                p.unlink()
            except OSError:
                pass
            continue
        try:
            pid, gb, root, *cmd = p.read_text().split("\t")
            out.append({"pid": int(pid), "gb": float(gb), "stack": root, "cmd": "\t".join(cmd).strip()})
        except (OSError, ValueError):
            pass
    return out


def acquire(stack, gb: float = None, budget_gb: float = None, poll: float = 5.0) -> None:
    """Block until a reservation of `gb` fits in the budget. Idempotent within a process."""
    global _held
    if _held is not None or os.environ.get("EDITSTACK_NO_SLOTS") == "1":
        return
    res = stack.cfg["resources"]
    gb = float(gb if gb is not None else res["per_process_gb"])
    budget = float(budget_gb if budget_gb is not None else res["ram_budget_gb"])
    d = _dir(stack)
    waited, t0 = False, time.time()
    while True:
        adm = os.open(d / "admit.lock", os.O_CREAT | os.O_RDWR, 0o666)
        fcntl.flock(adm, fcntl.LOCK_EX)
        try:
            used = sum(r["gb"] for r in reservations(stack))
            if used + gb <= budget:
                p = d / f"res_{os.getpid()}"
                fd = os.open(p, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o666)
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, f"{os.getpid()}\t{gb}\t{stack.root}\t{' '.join(sys.argv)[:300]}".encode())
                _held = (fd, p)
                if waited:
                    print(f"[slots] admitted after {time.time() - t0:.0f}s ({used:.0f}+{gb:.0f} of {budget:.0f} GB)",
                          file=sys.stderr, flush=True)
                return
        finally:
            fcntl.flock(adm, fcntl.LOCK_UN)
            os.close(adm)
        if not waited:
            print(f"[slots] waiting: {used:.0f} GB reserved + {gb:.0f} GB > budget {budget:.0f} GB", file=sys.stderr, flush=True)
            waited = True
        time.sleep(poll)


def release() -> None:
    global _held
    if _held is not None:
        fd, p = _held
        try:
            p.unlink()
        except OSError:
            pass
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        _held = None


def status(stack) -> str:
    rs = reservations(stack)
    res = stack.cfg["resources"]
    lines = [f"{r['pid']:>7d} {r['gb']:5.1f} GB  {Path(r['stack']).name:20s} {r['cmd'][:90]}" for r in rs]
    used = sum(r["gb"] for r in rs)
    lines.append(f"reserved {used:.0f} of {res['ram_budget_gb']} GB machine-wide; a new process of this stack needs "
                 f"{res['per_process_gb']} GB -> {int((res['ram_budget_gb'] - used) // res['per_process_gb'])} more fit now")
    return "\n".join(lines)
