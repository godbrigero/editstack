# editstack

Structured, evidence-driven **hand-editing of neural-network weights** (no gradient training), packaged so that an
agent (or a person) can pick up any model and edit it without guesswork.

**One stack = one editable model.** A stack is a folder with a fixed place for everything:

```
my-stack/
  AGENTS.md            the agent guide: phases, gates, rules, orchestration (CLAUDE.md points here)
  stack.json           task, metrics, qualification rules, RAM budget, harness version, python interpreter
  weights/original/    the model you supply (read-only)   weights/variants/  exports   weights/best -> promoted
  model/               decoded inventory, structural scan, ARCHITECTURE.md, MODEL_MAP.md, CLAIMS.json
  env/adapter.py       the only sim/model glue: load_policy(), run_episode()
  eval/suites/         fixed episode lists: screen / id / ood / heldout
  edits/<variant>/     edit.py (PARENT, OWNER, apply(W, ctx)) + data/ (fitted arrays)
  probes/  agents/     shared probes | one folder per sub-agent (BRIEF.md, REPORT.md, scratch)
  runs/  board/        eval results | LEADERBOARD.md, BEST_HISTORY.md
  docs/                LESSONS.md (evidence), EDIT_COOKBOOK.md, HARNESS_AUDIT.md
```

## Install (one command)

```bash
curl -LsSf https://raw.githubusercontent.com/godbrigero/editstack/main/install.sh | sh
```
This installs [uv](https://astral.sh/uv) if it's missing, then the `editstack` CLI. To install and create a stack in
one go:
```bash
curl -LsSf https://raw.githubusercontent.com/godbrigero/editstack/main/install.sh | sh -s -- \
    ~/stacks/my-model --weights /path/to/model --task "what to change" [--preset <name|folder>] [--python /venv/bin/python]
```
Alternatives: `uv tool install git+https://github.com/godbrigero/editstack`, or `pipx install git+...`. To put it
into the environment that runs your sim: `uv pip install --python /path/to/venv/bin/python git+https://github.com/godbrigero/editstack`.
Pin a version with `EDITSTACK_REF=<tag|commit>`.

The only dependency is numpy. torch and onnx are needed only for .pt/.bin/TorchScript or .onnx checkpoints. If the CLI
and your sim live in different environments, set `"python"` in stack.json (or pass `--python` to `new`). Commands that
load the model then re-run themselves under that interpreter automatically.

## Use

```bash
editstack new ~/stacks/my-model --weights /path/to/checkpoint --task "what it does and what to change" \
    [--python /path/to/sim/venv/bin/python] [--preset <name|folder>] [--link]
cd ~/stacks/my-model
```
Then tell your agent: **"Read AGENTS.md and work through it for this stack."**

`new` copies (or `--link`s) the weights into `weights/original/` and runs `decode`, which writes:
- `model/INVENTORY.md`: every tensor, its shape and dtype, repeated blocks collapsed, a guess at each role, stats.
- `model/SCAN.md`: structural-defect hints. On the SmolVLA checkpoint it flagged the padded action channels
  (`action_in_proj` cols [0:6] vs [6:32]) unprompted. That defect was the single biggest win of the original
  project.

Weight formats: safetensors (single or sharded; bf16 is exact; native, no torch needed), npz, `## name dtype
shape=[..]` text dumps, torch state_dict / TorchScript, ONNX initializers. Exports are written back in the same
format and verified bit-exact.

## Commands

```
editstack status | decode | verify
editstack edit new <v> --parent <p> --owner <me>     editstack apply <v>      (diff: tensors, values, rank, claims)
editstack eval <v> --suite <s> --seeds 0,1,2 --jobs 2 [--detach]   editstack wait <v> --suite <s>
editstack board | promote [--dry-run]                editstack export <v>
editstack agent new <name> --stage <s> --target "<numbers>"         editstack claims | slots
editstack run probes/x.py
```
Python: `from editstack import Stack, ops, fit` (`ops`: exact algebra, low-rank updates, gated SwiGLU circuits;
`fit`: ridge, `fit_delta`, `paired_map`, grouped CV R², gains).

## What it enforces (and why)

| mechanism | lesson it encodes |
|---|---|
| read-only original + in-memory edit chains + hash per chain | no hand-modified checkpoints; every number traceable to code |
| runs tagged with chain hash, harness version, suite hash; stale runs ignored and archived | a camera mismatch found on day 4 collapsed the best variant (6/40) until a camera-transfer stage; results silently mixed |
| paired seeds, Wilson intervals, sign test vs parent | two-seed results misled three times (e.g. 23/24 -> 31/36 at seed 3) |
| qualification rules + mandatory held-out suite before promotion | winner's curse from selecting dozens of variants on the same placements |
| machine-wide RAM slots (flock, crash-safe) | 15 model processes once hit ~110 GB |
| claims registry for neurons/channels | two agents once wired circuits into the same neurons |
| detach + wait + resumable evals | agents polled with sleep loops; network drops killed turns |
| owner briefs (`agent new`) | agents need a staged metric that can SEE their effect, a numeric target and a fixed suite; never two on one quantity |

## Presets (`editstack presets`)

- `toy`: a numpy "policy" with untrained padded inputs. It covers the full lifecycle in seconds and is used by the tests.
- Your own: any folder with `env/`, `eval/suites/`, `edits/`, `model/` and a partial `stack.json` (`--preset ./my-preset`).

## Tests

```bash
uv venv && uv pip install -e ".[dev]" && .venv/bin/python -m pytest -q
```
