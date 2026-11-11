"""A stack = one folder holding one editable model, its knowledge files, its sim adapter, edits, runs and leaderboard."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG = "stack.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "name": "",
    "task": "",
    "created": "",
    "weights": {"format": None, "original": "weights/original"},
    "python": None,
    "device": "cpu",
    "harness_version": "v1",
    "resources": {"ram_budget_gb": 60, "per_process_gb": 8, "lock_dir": "~/.cache/editstack/slots"},
    "metrics": {"primary": "success", "all": ["success"]},
    "qualify": {
        "rules": [
            {"suite": "ood", "min_seeds": 3, "min_episodes": 30},
            {"suite": "id", "min_seeds": 2, "min_episodes": 30, "min_rate": {"success": 0.9}},
        ],
        "rank": [["ood", "success"], ["id", "success"]],
        "require_heldout": True,
    },
    "real_hardware": {"allowed": False, "note": "Real runs only after sim qualification with a stated expected result."},
}


class StackError(RuntimeError):
    pass


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class Stack:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        cfg_path = self.root / CONFIG
        if not cfg_path.exists():
            raise StackError(f"{self.root} is not a stack (no {CONFIG}); run `editstack new` first")
        self.cfg = _merge(DEFAULT_CONFIG, json.loads(cfg_path.read_text()))

    # ------------------------------------------------------------------ discovery
    @classmethod
    def find(cls, start: Optional[Path] = None) -> "Stack":
        env = os.environ.get("EDITSTACK_ROOT")
        p = Path(env) if env else Path(start or Path.cwd())
        p = p.resolve()
        for q in [p, *p.parents]:
            if (q / CONFIG).exists():
                return cls(q)
        raise StackError(f"no {CONFIG} found in {p} or its parents (set EDITSTACK_ROOT or cd into a stack)")

    def save_cfg(self) -> None:
        (self.root / CONFIG).write_text(json.dumps(self.cfg, indent=2) + "\n")

    # ------------------------------------------------------------------ paths
    @property
    def original_dir(self) -> Path:
        return self.root / self.cfg["weights"]["original"]

    @property
    def variants_dir(self) -> Path:
        return self.root / "weights" / "variants"

    @property
    def best_link(self) -> Path:
        return self.root / "weights" / "best"

    @property
    def edits_dir(self) -> Path:
        return self.root / "edits"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def board_dir(self) -> Path:
        return self.root / "board"

    @property
    def suites_dir(self) -> Path:
        return self.root / "eval" / "suites"

    @property
    def harness_version(self) -> str:
        return str(self.cfg["harness_version"])

    # ------------------------------------------------------------------ suites
    def suites(self) -> Dict[str, dict]:
        out = {}
        for f in sorted(self.suites_dir.glob("*.json")):
            s = json.loads(f.read_text())
            s.setdefault("name", f.stem)
            s.setdefault("role", "selection")
            s["_path"] = str(f)
            out[s["name"]] = s
        return out

    def suite(self, name: str) -> dict:
        s = self.suites()
        if name not in s:
            raise StackError(f"unknown suite {name!r}; have {sorted(s)} (eval/suites/*.json)")
        return s[name]

    # ------------------------------------------------------------------ variants
    def variant_names(self) -> List[str]:
        return sorted(p.parent.name for p in self.edits_dir.glob("*/edit.py") if not p.parent.name.startswith("_"))

    def checkpoint(self):
        from . import formats
        return formats.load(self.original_dir, self.cfg["weights"].get("format"))

    def weights(self, variant: str = "original", check_claims: bool = True):
        """Original weights with the variant's edit chain applied in memory. Returns variant.Applied."""
        from .variant import apply_variant
        return apply_variant(self, variant, check_claims=check_claims)

    def policy(self, variant: str = "original", device: Optional[str] = None):
        """Acquire a RAM slot, apply the variant in memory and build a runnable policy via env/adapter.py."""
        from . import slots
        slots.acquire(self)
        applied = self.weights(variant)
        adapter = self.adapter()
        return adapter.load_policy(applied.tensors, self.adapter_ctx(device))

    def adapter(self):
        import importlib.util
        path = self.root / "env" / "adapter.py"
        if not path.exists():
            raise StackError("env/adapter.py is missing")
        if str(self.root / "env") not in sys.path:
            sys.path.insert(0, str(self.root / "env"))
        spec = importlib.util.spec_from_file_location("stack_adapter", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        for fn in ("load_policy", "run_episode"):
            if not hasattr(mod, fn):
                raise StackError(f"env/adapter.py must define {fn}()")
        return mod

    def adapter_ctx(self, device: Optional[str] = None, out_dir: Optional[Path] = None):
        from types import SimpleNamespace
        return SimpleNamespace(stack=self, root=self.root, original_dir=self.original_dir,
                               device=device or self.cfg.get("device", "cpu"), out_dir=out_dir,
                               harness_version=self.harness_version)
