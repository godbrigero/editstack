# __NAME__: edit stack

__TASK__

One folder = one editable model (built with [editstack](https://github.com/godbrigero/editstack)).

## Start here

**1. Install the tool (one command, any machine):**

```bash
curl -LsSf https://raw.githubusercontent.com/godbrigero/editstack/main/install.sh | sh
```

(Already have uv? Then `uv tool install git+https://github.com/godbrigero/editstack` does the same.)

**2. Put the model in.** Copy or symlink your model's weight files (safetensors, .pt/.bin, .onnx, .npz, or a text
dump) and its config/normaliser files into `weights/original/`, then decode them:

```bash
editstack add-weights /path/to/model      # skip if `editstack new --weights` already did it
editstack decode                          # -> model/INVENTORY.md, model/SCAN.md
```

**3. Hand it to an agent:**

> Read AGENTS.md and work through it for this stack. Goal: <what you want the model to do differently>.

The agent starts with Phase 0, a harness audit. In that phase it writes `env/adapter.py` (how to run the model
in your sim) and `eval/suites/` (the fixed test episodes), unless a preset already provided them.

## Everyday commands

```bash
editstack status                 # what's running, RAM, leaderboard top, deployed variant
editstack board                  # full leaderboard -> board/LEADERBOARD.md
editstack promote --dry-run      # what would become weights/best
```

Deployed weights: `weights/best` (a normal checkpoint folder in the original format).
History of what was promoted and why: `board/BEST_HISTORY.md`. Knowledge about the model: `model/`.

## Recreate this stack elsewhere

```bash
curl -LsSf https://raw.githubusercontent.com/godbrigero/editstack/main/install.sh | sh -s -- \
    ./__DIRNAME__ --weights /path/to/model __PRESET_FLAG__--task "__TASK_SHORT__"
```

Then copy this folder's `edits/`, `model/` and `agents/` across to bring the work with you. `weights/` and `runs/`
are not in git.
