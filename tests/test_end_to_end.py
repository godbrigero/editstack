"""End-to-end: new stack -> decode finds the padding defect -> edit -> claims -> eval (parallel) -> board -> promote."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "src" / "editstack" / "presets" / "toy"


def cli(*args, cwd, check=True):
    env = dict(os.environ, EDITSTACK_ROOT=str(cwd), PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-m", "editstack", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"editstack {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r


@pytest.fixture()
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))          # private slot dir
    w = tmp_path / "toy_weights"
    subprocess.run([sys.executable, str(TOY / "make_model.py"), str(w)], check=True, capture_output=True)
    st = tmp_path / "stack"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HOME=str(tmp_path / "home"))
    subprocess.run([sys.executable, "-m", "editstack", "new", str(st), "--weights", str(w), "--task", "reach target",
                    "--preset", "toy"], check=True, env=env, capture_output=True)
    assert not (st / "make_model.py").exists()                  # presets copy folders only
    os.environ["HOME"] = str(tmp_path / "home")
    return st


def test_lifecycle(stack):
    # decode found the untrained pad columns (bf16 storage) and wrote the knowledge files
    scan = (stack / "model" / "SCAN.md").read_text()
    assert "enc.weight: cols (inputs) split at 2" in scan, scan
    assert (stack / "model" / "ARCHITECTURE.md").exists()
    assert "intact" in cli("verify", cwd=stack).stdout
    # original is read-only
    assert not os.access(stack / "weights" / "original" / "model.safetensors", os.W_OK)

    # edit: disconnect pad inputs
    cli("edit", "new", "pad0", "--owner", "lead", cwd=stack)
    (stack / "edits" / "pad0" / "edit.py").write_text(
        "from editstack import ops\nPARENT=None\nOWNER='lead'\n"
        "def apply(W, ctx):\n    ops.zero_cols(W, 'enc.weight', slice(2, None))\n")
    out = cli("apply", "pad0", cwd=stack).stdout
    assert "enc.weight" in out and "changed tensors: 1" in out

    # claims: two variants outside one chain cannot take the same unit
    for name, parent in (("c1", "pad0"), ("c2", "pad0")):
        cli("edit", "new", name, "--parent", parent, cwd=stack)
        (stack / "edits" / name / "edit.py").write_text(
            f"PARENT='{parent}'\nOWNER='{name}'\n"
            "def apply(W, ctx):\n    W['head.weight'][:, 5] *= 1.0001\n    ctx.claim('head.weight', 1, [5], 'test')\n")
    cli("apply", "c1", cwd=stack)
    assert json.loads((stack / "model" / "CLAIMS.json").read_text()) == []      # a dry apply registers nothing
    cli("eval", "c1", "--suite", "screen", "--seeds", "0", "--episodes", "0", cwd=stack)   # eval registers c1's claims
    r = cli("apply", "c2", cwd=stack, check=False)
    assert r.returncode != 0 and "claim conflict" in (r.stderr + r.stdout)
    # an UNCLAIMED change of a claimed slice is also a conflict (derived from the diff)
    cli("edit", "new", "c3", "--parent", "pad0", cwd=stack)
    (stack / "edits" / "c3" / "edit.py").write_text("PARENT='pad0'\ndef apply(W, ctx):\n    W['head.weight'][:, 5] += 1\n")
    r = cli("apply", "c3", cwd=stack, check=False)
    assert r.returncode != 0 and "without claiming" in (r.stderr + r.stdout)
    for n in ("c1", "c2", "c3"):
        shutil.rmtree(stack / "edits" / n)

    # eval baseline and pad0 (parallel workers), paired seeds
    for v in ("original", "pad0"):
        cli("eval", v, "--suite", "id", "--seeds", "0,1", "--jobs", "2", cwd=stack)
        cli("eval", v, "--suite", "ood", "--seeds", "0,1,2", "--jobs", "2", cwd=stack)
    # resume: nothing to redo
    assert "nothing to do" in cli("eval", "pad0", "--suite", "id", "--seeds", "0", cwd=stack).stdout
    board = cli("board", cwd=stack).stdout
    assert "**pad0**" in board and "vs parent" in board
    # promotion requires held-out results
    assert "no held-out" in cli("promote", cwd=stack).stdout
    cli("eval", "pad0", "--suite", "heldout", "--seeds", "0", cwd=stack)
    pr = cli("promote", cwd=stack).stdout
    assert "weights/best -> pad0" in pr, pr
    best = stack / "weights" / "best"
    assert best.is_symlink() and (best / "MANIFEST.json").exists()

    # exported checkpoint is bit-exact: bf16 tensor round-trips, untouched tensors identical to original
    sys.path.insert(0, str(ROOT / "src"))
    from editstack import formats
    a = formats.load(stack / "weights" / "original")
    b = formats.load(best)
    assert b.dtypes["enc.weight"] == "BF16"
    assert np.array_equal(formats.to_storage(a.tensors["head.weight"], "F32"), formats.to_storage(b.tensors["head.weight"], "F32"))
    assert np.all(b.tensors["enc.weight"][:, 2:] == 0) and np.array_equal(b.tensors["enc.weight"][:, :2], a.tensors["enc.weight"][:, :2])

    # editing the edit makes old runs stale (archived on next eval, ignored by the board)
    p = stack / "edits" / "pad0" / "edit.py"
    p.write_text(p.read_text() + "\n# changed\n")
    board2 = cli("board", cwd=stack).stdout
    assert "stale" in board2


def test_textdump_roundtrip(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from editstack import formats
    f = tmp_path / "m.txt"
    f.write_text("# float32\n\n## L1.weight F32 shape=[2, 3]\n1 2 3\n4 5 6\n\n## L1.bias F32 shape=[2]\n0.5 -0.5\n")
    ck = formats.load(f)
    assert ck.fmt == "textdump" and ck.tensors["L1.weight"].shape == (2, 3)
    t = dict(ck.tensors)
    t["L1.weight"] = t["L1.weight"] * 2
    formats.save(ck, t, tmp_path / "out")
    back = formats.load(tmp_path / "out" / "m.txt")
    assert np.allclose(back.tensors["L1.weight"], [[2, 4, 6], [8, 10, 12]])


def test_bf16_rounding_exact():
    sys.path.insert(0, str(ROOT / "src"))
    from editstack.formats import bf16_to_f32, f32_to_bf16
    u = np.arange(0, 65536, dtype=np.uint32).astype(np.uint16)
    f = bf16_to_f32(u)
    ok = ~np.isnan(f)
    assert np.array_equal(f32_to_bf16(f[ok]), u[ok])           # every bf16 value round-trips


def test_knowledge_loop(stack):
    sdir = stack / "docs" / "strategies"
    sdir.mkdir(exist_ok=True)
    (sdir / "S99-test.md").write_text("# S99 test\n\nStatus: untested idea\n\n## Evidence log\n\n- seed line\n\n## Notes\nx\n")
    r = cli("learn", "pad0 fixed it: 0/36 -> 36/36", "--strategy", "S99", "--outcome", "win",
            "--evidence", "runs/pad0/ood", cwd=stack)
    assert "S99-test.md" in r.stdout
    txt = (sdir / "S99-test.md").read_text()
    assert txt.index("36/36") < txt.index("## Notes")                     # appended inside the evidence log
    assert "36/36" in (stack / "docs" / "LEARNINGS.md").read_text()
    assert cli("learn", "x", "--outcome", "win", cwd=stack, check=False).returncode != 0   # wins need evidence
    # local edit survives a sync; upstream copy lands beside it
    agents_md = stack / "AGENTS.md"
    agents_md.write_text(agents_md.read_text() + "\nLOCAL RULE\n")
    out = cli("docs", "sync", cwd=stack).stdout
    assert "CONFLICT AGENTS.md" in out and "LOCAL RULE" in agents_md.read_text()
    assert (stack / "AGENTS.md.upstream").exists()
    patch = cli("docs", "contribute", cwd=stack).stdout
    p = Path(patch.split("wrote ")[1].split("\n")[0])
    body = p.read_text()
    assert "LOCAL RULE" in body and "S99-test.md" in body and "36/36" in body



def test_reviewer_regressions(stack):
    """Bugs found by the independent review, each reproduced on a toy stack before the fix."""
    (stack / "env" / "helper.py").write_text("ZERO_FROM = 2\n")
    cli("edit", "new", "pz", cwd=stack)
    (stack / "edits" / "pz" / "edit.py").write_text(
        "import helper\nfrom editstack import ops\nPARENT=None\n"
        "def apply(W, ctx):\n    ops.zero_cols(W, 'enc.weight', slice(helper.ZERO_FROM, None))\n")
    cli("eval", "pz", "--suite", "ood", "--seeds", "0", cwd=stack)
    # 1. chain hash covers env helpers: changing the helper makes old runs stale
    (stack / "env" / "helper.py").write_text("ZERO_FROM = 3\n")
    assert "1 stale" in cli("board", cwd=stack).stdout
    out = cli("eval", "pz", "--suite", "ood", "--seeds", "0", cwd=stack).stdout
    assert "archived stale" in out and "weights changed" in out
    # 2. partial seeds never count
    cli("eval", "pz", "--suite", "ood", "--seeds", "1", "--episodes", "0-3", cwd=stack)
    board = cli("board", cwd=stack).stdout
    assert "1 seeds, 1 partial" in board, board
    # the rerun really used ZERO_FROM = 3 (no stale bytecode): channel 2 still carries noise, so not every episode passes
    assert "success 12/12" not in board.split("**pz**")[1].split("\n")[0], board
    # 3. a corrupt (truncated) results line is skipped, not fatal
    with open(stack / "runs" / "pz" / "ood" / "s0" / "results.jsonl", "a") as f:
        f.write('{"episode": 99, "metr')
    cli("board", cwd=stack)
    cli("status", cwd=stack)
    # 4. worker init failure aborts quickly instead of respawning forever
    good = (stack / "env" / "adapter.py").read_text()
    (stack / "env" / "adapter.py").write_text(good.replace("def load_policy(tensors, ctx):",
                                                          "def load_policy(tensors, ctx):\n    raise RuntimeError('boom')"))
    r = cli("eval", "pz", "--suite", "id", "--seeds", "0", "--jobs", "2", cwd=stack, check=False)
    assert r.returncode != 0 and "aborting" in (r.stdout + r.stderr) and (r.stdout + r.stderr).count("boom") < 50
    (stack / "env" / "adapter.py").write_text(good)
    # 5. detach + wait waits for THIS job (no race with an older DONE)
    cli("eval", "pz", "--suite", "id", "--seeds", "0,1", "--detach", cwd=stack)
    out = cli("wait", "pz", "--suite", "id", cwd=stack).stdout
    assert out.count("full_suite") == 2, out
    # 6. a second eval of a dir that is being evaluated is refused
    sys.path.insert(0, str(ROOT / "src"))
    from editstack.evaluate import lock_dir
    fd = lock_dir(stack / "runs" / "pz" / "id" / "s0")
    r = cli("eval", "pz", "--suite", "id", "--seeds", "0", "--fresh", cwd=stack, check=False)
    os.close(fd)
    assert r.returncode != 0 and "already being evaluated" in (r.stdout + r.stderr)
    # 7. ablation variants work and compare against the full chain
    cli("edit", "new", "pz2", "--parent", "pz", cwd=stack)
    (stack / "edits" / "pz2" / "edit.py").write_text("PARENT='pz'\ndef apply(W, ctx):\n    W['head.bias'][0] += 0.0\n    W['head.bias'][1] += 0.001\n")
    out = cli("ablate", "pz2", "--suite", "screen", "--seeds", "0", cwd=stack).stdout
    assert "pz2~minus~pz" in out and "pz2~minus~pz2" in out
    # 8. original weights are protected: a changed file is caught on the next eval
    w = stack / "weights" / "original" / "config.json"
    os.chmod(w, 0o644)
    w.write_text(w.read_text() + " ")
    r = cli("eval", "pz", "--suite", "screen", "--seeds", "0", cwd=stack, check=False)
    assert r.returncode != 0 and "ORIGINAL WEIGHTS CHANGED" in (r.stdout + r.stderr)


def test_qualify_ladder(stack):
    cli("edit", "new", "q0", cwd=stack)
    (stack / "edits" / "q0" / "edit.py").write_text(
        "from editstack import ops\nPARENT=None\ndef apply(W, ctx):\n    ops.zero_cols(W, 'enc.weight', slice(2, None))\n")
    out = cli("qualify", "q0", cwd=stack).stdout
    assert "rung c passed" in out and "rung d" in out
    assert "weights/best -> q0" in cli("promote", cwd=stack).stdout
    # a harmful edit stops at the paired early-stop rung
    cli("edit", "new", "bad", "--parent", "q0", cwd=stack)
    (stack / "edits" / "bad" / "edit.py").write_text("PARENT='q0'\ndef apply(W, ctx):\n    W['head.bias'][:] += 1.0\n")
    r = cli("qualify", "bad", cwd=stack, check=False)
    assert r.returncode != 0 and "paired early stop" in (r.stdout + r.stderr)


def test_audit2_regressions(stack):
    # adding a probe-only helper to the adapter does not invalidate runs
    cli("edit", "new", "z", cwd=stack)
    (stack / "edits" / "z" / "edit.py").write_text("PARENT=None\ndef apply(W, ctx):\n    W['head.bias'][0] += 0.0\n    W['head.bias'][1] += 1e-3\n")
    cli("eval", "z", "--suite", "screen", "--seeds", "0", cwd=stack)
    ad = stack / "env" / "adapter.py"
    ad.write_text(ad.read_text() + '\n\ndef capture(policy, obs, names):\n    """probe helper"""\n    return {}\n')
    assert "stale" not in cli("board", cwd=stack).stdout.split("**z**")[1].split("\n")[0]
    # but changing run_episode does
    ad.write_text(ad.read_text().replace("err < 0.08}", "err < 0.0801}"))
    assert "stale" in cli("board", cwd=stack).stdout.split("**z**")[1].split("\n")[0]
    # early reservation blocks a second variant from taking the same units
    cli("edit", "new", "z2", cwd=stack)
    cli("claims", "reserve", "z", "head.weight", "1", "7,8", "--purpose", "test", cwd=stack)
    (stack / "edits" / "z2" / "edit.py").write_text("PARENT=None\ndef apply(W, ctx):\n    W['head.weight'][:, 7] += 1\n")
    r = cli("apply", "z2", cwd=stack, check=False)
    assert r.returncode != 0 and "claimed by z" in (r.stdout + r.stderr)
    # docs sync merges a strategy's local evidence lines instead of conflicting
    sdir = stack / "docs" / "strategies"
    f = next(sdir.glob("S01-*.md"))
    cli("learn", "unit-test line", "--strategy", "S01", "--outcome", "info", cwd=stack)
    out = cli("docs", "sync", cwd=stack).stdout
    assert "CONFLICT" not in out and "unit-test line" in f.read_text()
