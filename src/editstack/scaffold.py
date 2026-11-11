"""Create stacks, edits and agent folders from the packaged templates."""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Optional

TEMPLATE = Path(__file__).parent / "template"
PRESETS = Path(__file__).parent / "presets"


def list_presets():
    return sorted(p.name for p in PRESETS.iterdir() if p.is_dir() and not p.name.startswith("_"))


def resolve_preset(preset) -> Path:
    """A preset is a bundled name (editstack new --preset smolvla-so101) or a folder path."""
    p = Path(preset).expanduser()
    if p.is_dir():
        return p.resolve()
    if (PRESETS / str(preset)).is_dir():
        return (PRESETS / str(preset)).resolve()
    raise SystemExit(f"unknown preset {preset!r}; bundled: {list_presets()} (or pass a folder)")


def _copy_tree(src: Path, dst: Path) -> None:
    for p in sorted(src.rglob("*")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(src)
        rel = Path(str(rel).replace("dot.gitignore", ".gitignore"))
        t = dst / rel
        if p.is_dir():
            t.mkdir(parents=True, exist_ok=True)
        elif not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)


def new_stack(dest: Path, weights: Optional[Path], name: str = "", task: str = "", link: bool = False,
              python: Optional[str] = None, decode: bool = True, preset: Optional[Path] = None) -> Path:
    from .stack import DEFAULT_CONFIG, Stack
    dest = Path(dest).expanduser().resolve()
    if (dest / "stack.json").exists():
        raise SystemExit(f"{dest} is already a stack")
    dest.mkdir(parents=True, exist_ok=True)
    _copy_tree(TEMPLATE / "stack", dest)
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg.update(name=name or dest.name, task=task, created=time.strftime("%Y-%m-%d %H:%M"), python=python)
    if preset:
        preset = resolve_preset(preset)
        from .stack import _merge
        if (preset / "stack.json").exists():
            cfg = _merge(cfg, json.loads((preset / "stack.json").read_text()))
        if (preset / "eval" / "suites").exists():      # preset suites replace the placeholder ones
            for f in (dest / "eval" / "suites").glob("*.json"):
                f.unlink()
        for p in sorted(preset.rglob("*")):
            rel = p.relative_to(preset)
            if p.is_dir() or "__pycache__" in p.parts or len(rel.parts) == 1:   # presets ship folders (+ stack.json)
                continue
            t = dest / rel
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)
        cfg["preset"] = str(preset)
    (dest / "stack.json").write_text(json.dumps(cfg, indent=2) + "\n")
    _render_readme(dest, cfg, preset)
    orig = dest / "weights" / "original"
    orig.mkdir(parents=True, exist_ok=True)
    if weights:
        add_weights(dest, Path(weights), link=link)
    if shutil.which("git") and not (dest / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=dest)
    st = Stack(dest)
    if weights and decode:
        from .decode import decode as run_decode
        run_decode(st)
    return dest


def _render_readme(dest: Path, cfg: dict, preset) -> None:
    readme = dest / "README.md"
    if not readme.exists() or "__NAME__" not in readme.read_text():
        return
    task = cfg.get("task") or "(task not set: put one sentence about what the model does and what to change in stack.json)"
    short = (cfg.get("task") or "what to change").replace('"', "'")
    pname = Path(preset).name if preset else ""
    txt = (readme.read_text().replace("__NAME__", cfg.get("name") or dest.name).replace("__TASK_SHORT__", short)
           .replace("__TASK__", task).replace("__DIRNAME__", dest.name)
           .replace("__PRESET_FLAG__", f"--preset {pname} " if pname else ""))
    readme.write_text(txt)


def add_weights(dest: Path, weights: Path, link: bool = False) -> None:
    weights = weights.expanduser().resolve()
    orig = Path(dest) / "weights" / "original"
    orig.mkdir(parents=True, exist_ok=True)
    files = [weights] if weights.is_file() else [p for p in sorted(weights.iterdir()) if p.is_file() and not p.name.startswith(".")]
    for f in files:
        t = orig / f.name
        if t.exists() or t.is_symlink():
            raise SystemExit(f"{t} already exists; weights/original is write-once")
        if link:
            t.symlink_to(f)
        else:
            shutil.copy2(f, t)
            t.chmod(t.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def new_edit(stack, name: str, parent: Optional[str], owner: str, desc: str) -> Path:
    d = stack.edits_dir / name
    if (d / "edit.py").exists():
        raise SystemExit(f"edits/{name} already exists")
    if parent and parent != "original" and not (stack.edits_dir / parent / "edit.py").exists():
        raise SystemExit(f"parent edits/{parent} does not exist")
    (d / "data").mkdir(parents=True, exist_ok=True)
    tpl = (TEMPLATE / "edit.py.tpl").read_text()
    (d / "edit.py").write_text(tpl.format(name=name, parent=repr(parent if parent and parent != "original" else None),
                                          owner=owner, desc=desc or "TODO: hypothesis, evidence, expected effect, exactness."))
    return d


def new_agent(stack, name: str, stage: str, target: str, suites: str, parent: str, budget_procs: int) -> Path:
    d = stack.root / "agents" / name
    if d.exists():
        raise SystemExit(f"agents/{name} already exists")
    d.mkdir(parents=True)
    tpl = (TEMPLATE / "BRIEF.md.tpl").read_text()
    (d / "BRIEF.md").write_text(tpl.format(name=name, stage=stage, target=target, suites=suites, parent=parent,
                                           procs=budget_procs, root=stack.root, task=stack.cfg.get("task", ""),
                                           date=time.strftime("%Y-%m-%d")))
    (d / "REPORT.md").write_text(f"# {name}: report\n\nStatus: started {time.strftime('%Y-%m-%d %H:%M')}\n\n"
                                 "## Result (one line, numbers)\n\n## What I changed (variants, tensors, claims)\n\n"
                                 "## Evidence (run dirs, probe outputs)\n\n## Negative results\n\n"
                                 "## Facts learned about the model (also appended to model/MODEL_MAP.md)\n\n## Next\n")
    return d
