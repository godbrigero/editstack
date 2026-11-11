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
    r = cli("apply", "c2", cwd=stack, check=False)
    assert r.returncode != 0 and "claim conflict" in (r.stderr + r.stdout)

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
