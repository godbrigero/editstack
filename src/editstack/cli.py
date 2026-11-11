"""editstack command line. Run `editstack -h` or `editstack <command> -h`."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# commands that load the model / sim and must run under the stack's python (stack.json "python")
RUNTIME_CMDS = {"eval", "export", "promote", "apply", "decode", "verify", "run"}


def _stack():
    from .stack import Stack
    return Stack.find()


def _maybe_reexec(cmd: str, argv) -> None:
    if cmd not in RUNTIME_CMDS or os.environ.get("EDITSTACK_REEXEC") == "1":
        return
    try:
        st = _stack()
    except Exception:
        return
    py = st.cfg.get("python")
    if not py:
        return
    py = os.path.expanduser(py)
    # compare venv paths, NOT resolved binaries: two venvs often symlink to the same base interpreter
    if os.path.abspath(py) == os.path.abspath(sys.executable):
        return
    if not Path(py).exists():
        raise SystemExit(f"stack.json python={py} does not exist")
    env = dict(os.environ, EDITSTACK_REEXEC="1", EDITSTACK_ROOT=str(st.root),
               PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parent.parent), os.environ.get("PYTHONPATH", "")]))
    os.execve(py, [py, "-m", "editstack", *argv], env)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="editstack", description=__doc__,
                                epilog="Start: editstack new ./my-stack --weights /path/to/model --task '...'; then read AGENTS.md")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("new", help="create a stack folder (template + weights + decode)")
    a.add_argument("dest")
    a.add_argument("--weights", help="model file or folder to place in weights/original")
    a.add_argument("--task", default="", help="one sentence: what the model does and what you want changed")
    a.add_argument("--name", default="")
    a.add_argument("--link", action="store_true", help="symlink weight files instead of copying")
    a.add_argument("--python", help="interpreter that has the model/sim dependencies (eval runs under it)")
    a.add_argument("--no-decode", action="store_true")
    a.add_argument("--preset", help="folder with model-family files (env/, eval/suites/, edits/, model/, stack.json overrides)")

    a = sub.add_parser("add-weights", help="put weights into weights/original of the current stack (write-once)")
    a.add_argument("path")
    a.add_argument("--link", action="store_true")

    sub.add_parser("decode", help="inventory + structural scan of weights/original into model/")
    sub.add_parser("verify", help="check weights/original still matches model/ORIGINAL.sha256.json")
    sub.add_parser("status", help="one-shot digest: running evals, RAM slots, leaderboard top, best")

    a = sub.add_parser("edit", help="edit scaffolding")
    es = a.add_subparsers(dest="sub", required=True)
    b = es.add_parser("new", help="create edits/<name>/edit.py")
    b.add_argument("name")
    b.add_argument("--parent", default="original")
    b.add_argument("--owner", default="")
    b.add_argument("--desc", default="")
    es.add_parser("list", help="list variants and their chains")

    a = sub.add_parser("agent", help="agent scaffolding")
    ags = a.add_subparsers(dest="sub", required=True)
    b = ags.add_parser("new", help="create agents/<name>/BRIEF.md + REPORT.md for a stage owner")
    b.add_argument("name")
    b.add_argument("--stage", required=True, help="the one benchmark stage this agent owns, e.g. 'reach'")
    b.add_argument("--target", required=True, help="numeric target, e.g. 'ood reach >= 34/36 with id reach >= 90%%'")
    b.add_argument("--suites", default="screen,ood,id")
    b.add_argument("--parent", default="original", help="variant to build on")
    b.add_argument("--procs", type=int, default=2, help="max concurrent model processes for this agent")

    a = sub.add_parser("apply", help="apply a variant in memory and print the weight diff (no files written)")
    a.add_argument("variant")

    a = sub.add_parser("export", help="bake a variant into weights/variants/<name> (verified bit-exact)")
    a.add_argument("variant")
    a.add_argument("--out")

    a = sub.add_parser("eval", help="closed-loop eval of a variant on a suite")
    a.add_argument("variant")
    a.add_argument("--suite", required=True)
    a.add_argument("--seeds", default="0", help="e.g. 0,1,2")
    a.add_argument("--episodes", help="subset, e.g. 0-5 or 1,4")
    a.add_argument("--jobs", type=int, default=1, help="parallel model processes (each takes a RAM slot)")
    a.add_argument("--device")
    a.add_argument("--fresh", action="store_true", help="archive existing results and rerun")
    a.add_argument("--detach", action="store_true", help="run in a new session; prints the log path; use `editstack wait`")
    a.add_argument("--note", default="")

    a = sub.add_parser("wait", help="block until evals of a variant (and suite) have DONE markers; print summary")
    a.add_argument("variant")
    a.add_argument("--suite")
    a.add_argument("--timeout", type=float, default=7200)

    sub.add_parser("board", help="print and write board/LEADERBOARD.md")
    a = sub.add_parser("promote", help="point weights/best at the top qualified variant (exports if needed)")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--force", action="store_true", help="promote without held-out results")
    a.add_argument("--variant")

    sub.add_parser("slots", help="machine-wide RAM reservations")
    sub.add_parser("claims", help="list claimed neurons/channels")
    sub.add_parser("guide", help="print the path of the agent guide (AGENTS.md)")

    a = sub.add_parser("run", help="run a python script with the stack on sys.path under the stack's python")
    a.add_argument("script")
    a.add_argument("args", nargs=argparse.REMAINDER)

    if argv and argv[0] in RUNTIME_CMDS:
        _maybe_reexec(argv[0], argv)
    args = p.parse_args(argv)
    try:
        _dispatch(args, argv)
    except Exception as e:  # concise errors for agents
        from .stack import StackError
        from .variant import EditError
        from .claims import ClaimConflict
        if isinstance(e, (StackError, EditError, ClaimConflict)):
            raise SystemExit(f"error: {e}")
        raise


def _dispatch(args, argv) -> None:
    from . import scaffold
    cmd = args.cmd
    if cmd == "new":
        d = scaffold.new_stack(Path(args.dest), Path(args.weights) if args.weights else None, args.name, args.task,
                               args.link, args.python, decode=not args.no_decode,
                               preset=Path(args.preset) if args.preset else None)
        print(f"stack created: {d}")
        if not args.weights:
            print("next: editstack add-weights /path/to/model && editstack decode")
        print("next: read AGENTS.md (Phase 0) - fill env/adapter.py, eval/suites/, model/ARCHITECTURE.md")
        return
    st = _stack()
    if cmd == "add-weights":
        scaffold.add_weights(st.root, Path(args.path), link=args.link)
        print("added; now run: editstack decode")
    elif cmd == "decode":
        from .decode import decode
        r = decode(st)
        inv, h = r["inventory"], r["hints"]
        print(f"{inv['format']}: {inv['n_tensors']} tensors, {inv['n_params']:,} params -> model/INVENTORY.md, model/SCAN.md")
        for k, v in h.items():
            if v:
                print(f"  {k}: {len(v)} hint(s)" + (f"  e.g. {v[0][:150]}" if k in ("slices", "nonfinite", "uniform_like") else ""))
    elif cmd == "verify":
        from .formats import sha256_file
        ref = json.loads((st.model_dir / "ORIGINAL.sha256.json").read_text())
        bad = [n for n, h in ref.items() if sha256_file(st.original_dir / n) != h]
        print("original weights intact" if not bad else f"ORIGINAL WEIGHTS CHANGED: {bad}")
        if bad:
            raise SystemExit(1)
    elif cmd == "edit":
        from .variant import chain_of
        if args.sub == "new":
            d = scaffold.new_edit(st, args.name, args.parent, args.owner, args.desc)
            print(f"created {d.relative_to(st.root)}/edit.py")
        else:
            for v in st.variant_names():
                print(f"{v:24s} {' -> '.join(['original', *chain_of(st, v)])}")
    elif cmd == "agent":
        d = scaffold.new_agent(st, args.name, args.stage, args.target, args.suites, args.parent, args.procs)
        print(f"created {d.relative_to(st.root)}/BRIEF.md - delegate with: 'Read {d}/BRIEF.md and AGENTS.md, then do it.'")
    elif cmd == "apply":
        from .variant import apply_variant, diff_report
        print(diff_report(apply_variant(st, args.variant)))
    elif cmd == "export":
        from .variant import export
        out = export(st, args.variant, Path(args.out) if args.out else None)
        print(f"exported and verified: {out}")
    elif cmd == "eval":
        from .evaluate import evaluate, detach
        if args.detach:
            log = detach(st, argv)
            print(f"detached. log: {log}\nwait with: editstack wait {args.variant} --suite {args.suite}")
            return
        seeds = [int(s) for s in args.seeds.split(",") if s != ""]
        evaluate(st, args.variant, args.suite, seeds, args.episodes, args.jobs, args.device, args.fresh, args.note)
        from .board import write
        write(st)
    elif cmd == "wait":
        _wait(st, args.variant, args.suite, args.timeout)
    elif cmd == "board":
        from .board import write
        print(write(st))
    elif cmd == "promote":
        from .board import promote
        print(promote(st, args.dry_run, args.force, args.variant))
    elif cmd == "slots":
        from .slots import status
        print(status(st))
    elif cmd == "claims":
        from .claims import load
        for c in load(st):
            print(f"{c['variant']:16s} {c['owner']:10s} {c['tensor']} axis {c['axis']} idx {c['indices'][:16]} - {c['purpose']}")
    elif cmd == "guide":
        print(st.root / "AGENTS.md")
    elif cmd == "status":
        _status(st)
    elif cmd == "run":
        sys.path[:0] = [str(st.root), str(st.root / "env")]
        sys.argv = [args.script, *args.args]
        import runpy
        runpy.run_path(args.script, run_name="__main__")


def _running(st):
    out = []
    for m in st.runs_dir.glob("*/*/s*/meta.json"):
        if m.parent.parent.parent.name.startswith("_") or (m.parent / "DONE").exists():
            continue
        meta = json.loads(m.read_text())
        pid = meta.get("pid")
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                pass
        n = sum(1 for _ in open(m.parent / "results.jsonl")) if (m.parent / "results.jsonl").exists() else 0
        out.append((m.parent.relative_to(st.root), n, len(meta["episodes_planned"]), alive))
    return out


def _status(st) -> None:
    from .slots import status as slot_status
    from .board import summarize
    print(f"stack {st.cfg['name']} ({st.root})  harness {st.harness_version}")
    print(f"task: {st.cfg.get('task')}")
    run = _running(st)
    print(f"\nunfinished eval dirs: {len(run)}")
    for d, n, tot, alive in run:
        print(f"  {d}  {n}/{tot}  {'running' if alive else 'NOT RUNNING (dead/interrupted: rerun same command to resume)'}")
    print("\nRAM slots:\n" + "\n".join("  " + l for l in slot_status(st).splitlines()))
    b = summarize(st)
    print("\nleaderboard top 5 (editstack board for all):")
    p = st.cfg["metrics"]["primary"]
    for r in b[:5]:
        cells = ", ".join(f"{s} {x[p]}/{x['n']}" for s, x in r["suites"].items() if x["n"])
        print(f"  {'Q' if r['qualified'] else ' '} {r['variant']:20s} {cells}")
    print(f"\nweights/best -> {os.readlink(st.best_link) if st.best_link.is_symlink() else '(none)'}")


def _wait(st, variant, suite, timeout) -> None:
    t0 = time.time()
    pat = f"{variant}/{suite or '*'}/s*"
    while True:
        dirs = [d for d in st.runs_dir.glob(pat) if (d / "meta.json").exists()]
        if dirs and all((d / "DONE").exists() for d in dirs):
            for d in sorted(dirs):
                print(f"{d.relative_to(st.root)}: {(d / 'DONE').read_text().strip()}")
            return
        alive = [x for x in _running(st) if str(x[0]).startswith(f"runs/{variant}/") and x[3]]
        if dirs and not alive and time.time() - t0 > 60:
            raise SystemExit(f"no live process is producing {pat} and not all DONE - rerun the eval command to resume")
        if time.time() - t0 > timeout:
            raise SystemExit(f"timeout after {timeout:.0f}s waiting for {pat}")
        time.sleep(10)


if __name__ == "__main__":
    main()
