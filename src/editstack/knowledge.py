"""The evolving knowledge base: record learnings, sync guides from upstream, contribute learnings back.

Loop:
  1. an agent finishes an experiment -> `editstack learn ...` appends a dated, evidenced line to
     docs/LEARNINGS.md (journal) and, with --strategy, to that strategy's "Evidence log".
  2. new editstack versions ship improved guides -> `editstack docs sync` updates a stack's docs/ and AGENTS.md.
     Files the stack changed locally are never overwritten: the upstream version lands next to them as
     <file>.upstream for the agent to merge. LEARNINGS.md is never touched.
  3. `editstack docs contribute` writes a patch of the stack's learnings and edited guides against the
     template, for a PR to the editstack repo (ask the user before opening a PR: it publishes).
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional

from .scaffold import TEMPLATE

GUIDE_FILES = ["AGENTS.md"]          # plus everything under docs/
NEVER_SYNC = {"docs/LEARNINGS.md"}
VERSIONS = "docs/.versions.json"
OUTCOMES = ("win", "loss", "neutral", "mixed", "info")


def _h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _upstream_files() -> List[str]:
    root = TEMPLATE / "stack"
    files = [f for f in GUIDE_FILES if (root / f).exists()]
    files += sorted(str(p.relative_to(root)) for p in (root / "docs").rglob("*") if p.is_file() and p.name != ".versions.json")
    return [f for f in files if f not in NEVER_SYNC]


def record_versions(stack_root: Path) -> None:
    """Remember the hash of every guide as installed, so later syncs can tell local edits from upstream ones."""
    vers = {f: _h(stack_root / f) for f in _upstream_files() if (stack_root / f).exists()}
    (stack_root / VERSIONS).write_text(json.dumps(vers, indent=1) + "\n")


# ---------------------------------------------------------------------------------------------------- learn


def strategy_file(stack, sid: str) -> Path:
    d = stack.root / "docs" / "strategies"
    hits = sorted(d.glob(f"{sid}-*.md")) + sorted(d.glob(f"{sid}.md"))
    if not hits:
        have = sorted(p.stem.split("-")[0] for p in d.glob("S*.md"))
        raise SystemExit(f"no strategy {sid} in docs/strategies (have: {have})")
    return hits[0]


def learn(stack, text: str, strategy: Optional[str] = None, outcome: str = "info", evidence: str = "",
          agent: str = "", variant: str = "") -> List[Path]:
    if outcome not in OUTCOMES:
        raise SystemExit(f"--outcome must be one of {OUTCOMES}")
    if outcome in ("win", "loss", "mixed") and not evidence:
        raise SystemExit("a win/loss/mixed learning needs --evidence (run dir, probe output, board line)")
    date = time.strftime("%Y-%m-%d")
    model = stack.cfg.get("name") or stack.root.name
    line = (f"- {date} [{model}{'/' + variant if variant else ''}]{' (' + agent + ')' if agent else ''} "
            f"**{outcome}**: {text.strip()}" + (f" (evidence: {evidence})" if evidence else ""))
    written = []
    journal = stack.root / "docs" / "LEARNINGS.md"
    if not journal.exists():
        journal.write_text(LEARNINGS_HEADER)
    with open(journal, "a") as f:
        f.write(line.replace("- ", f"- [{strategy}] ", 1) + "\n" if strategy else line + "\n")
    written.append(journal)
    if strategy:
        sf = strategy_file(stack, strategy)
        txt = sf.read_text()
        m = re.search(r"^#+\s*Evidence log.*$", txt, flags=re.M | re.I)
        if m:
            # append at the end of the evidence-log section (before the next heading)
            nxt = re.search(r"^#+\s", txt[m.end():], flags=re.M)
            pos = m.end() + nxt.start() if nxt else len(txt)
            txt = txt[:pos].rstrip("\n") + "\n" + line + "\n" + ("\n" + txt[pos:].lstrip("\n") if nxt else "")
        else:
            txt = txt.rstrip("\n") + "\n\n## Evidence log\n\n" + line + "\n"
        sf.write_text(txt)
        written.append(sf)
    return written


LEARNINGS_HEADER = """# Learnings journal (this stack)

Append with `editstack learn "<what happened, with numbers>" --outcome win|loss|neutral|mixed|info
--evidence <path> [--strategy Sxx] [--variant v] [--agent name]`. One line per experiment or insight.
General lessons that should change a guide or rule: write them here first with evidence, then propose the
guide change (edit the guide in this stack and run `editstack docs contribute`). Never delete lines.

"""


# ---------------------------------------------------------------------------------------------------- sync


def sync(stack, dry_run: bool = False) -> List[str]:
    root = stack.root
    up_root = TEMPLATE / "stack"
    vfile = root / VERSIONS
    installed = json.loads(vfile.read_text()) if vfile.exists() else {}
    report = []
    for rel in _upstream_files():
        up, loc = up_root / rel, root / rel
        if not loc.exists():
            report.append(f"new      {rel}")
            if not dry_run:
                loc.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(up, loc)
                installed[rel] = _h(loc)
            continue
        if _h(up) == _h(loc):
            installed[rel] = _h(loc)
            continue
        locally_edited = installed.get(rel) is not None and installed[rel] != _h(loc)
        unknown = rel not in installed
        if locally_edited or unknown:
            report.append(f"CONFLICT {rel}: edited locally -> upstream written to {rel}.upstream (merge, keep the evidence logs)")
            if not dry_run:
                shutil.copy2(up, Path(str(loc) + ".upstream"))
        else:
            report.append(f"updated  {rel}")
            if not dry_run:
                shutil.copy2(up, loc)
                installed[rel] = _h(loc)
    if not dry_run:
        vfile.parent.mkdir(parents=True, exist_ok=True)
        vfile.write_text(json.dumps(installed, indent=1) + "\n")
    return report or ["guides already up to date"]


# ---------------------------------------------------------------------------------------------------- contribute


def contribute(stack, out: Optional[Path] = None) -> Path:
    """Unified diff of this stack's guides vs the installed template + its learnings journal."""
    up_root = TEMPLATE / "stack"
    chunks = []
    for rel in _upstream_files():
        loc = stack.root / rel
        up = up_root / rel
        if loc.exists() and up.exists() and _h(loc) != _h(up):
            chunks += difflib.unified_diff(up.read_text().splitlines(True), loc.read_text().splitlines(True),
                                           f"a/src/editstack/template/stack/{rel}", f"b/src/editstack/template/stack/{rel}")
    for p in sorted((stack.root / "docs").rglob("*.md")):
        rel = str(p.relative_to(stack.root))
        if not (up_root / rel).exists() and rel not in NEVER_SYNC:     # brand-new guide or strategy written in the stack
            chunks += difflib.unified_diff([], p.read_text().splitlines(True), "/dev/null", f"b/src/editstack/template/stack/{rel}")
    out = Path(out) if out else stack.root / "board" / f"knowledge-{time.strftime('%Y%m%d-%H%M')}.patch"
    out.parent.mkdir(parents=True, exist_ok=True)
    journal = stack.root / "docs" / "LEARNINGS.md"
    header = (f"# editstack knowledge contribution from stack '{stack.cfg.get('name')}' ({time.strftime('%Y-%m-%d')})\n"
              f"# Learnings journal follows as comments; apply the diff with `git apply` in the editstack repo.\n")
    if journal.exists():
        header += "".join("# " + l for l in journal.read_text().splitlines(True))
    out.write_text(header + "\n" + "".join(chunks))
    return out
