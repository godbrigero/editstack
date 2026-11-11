# Agent guide: editing this model's weights

You are editing the weights of ONE model (this stack) by hand: no gradient training, no new data, unless the user
explicitly says otherwise. This guide is the distilled result of ~5 days and ~50 agent sessions of hand-editing a
SmolVLA robot policy (see docs/LESSONS.md for the evidence behind every rule). Follow it in order. It is fast
because it removes guesswork: every edit is derived, every claim is measured, every number is comparable.

Read first: `stack.json` (task, metrics, qualification rules), `model/ARCHITECTURE.md`, `model/MODEL_MAP.md`,
`board/LEADERBOARD.md`, and `editstack status`.

---------------------------------------------------------------------------------------------------------------

## 0. Ground rules (non-negotiable)

1. **`weights/original/` is read-only.** Never write to it. Every variant is `original + edit chain`, applied in
   memory. `editstack verify` must always pass.
2. **Every edit is code in `edits/<variant>/edit.py`** (+ fitted arrays in `edits/<variant>/data/`). No edits
   in notebooks, probes or one-off scripts. No hand-modified checkpoint files. One variant = one directory.
3. **Scripts live in exactly three places:** `env/` (the sim/model adapter, owned by the lead), `probes/` (shared,
   reusable probes), `agents/<name>/` (an agent's own scratch scripts and data). Nothing at the stack root.
4. **Never state a result you did not measure.** Every number in a report cites a run dir, a probe output file or
   a command. "Should work" is not a result.
5. **Only the leaderboard decides.** A variant is better only if `editstack board` says so under the
   qualification rules. Do not promote on 2 seeds, on a screen, or on a feeling.
6. **RAM: every model process goes through `editstack` (eval/policy) which takes a machine-wide slot.** Never
   load the model outside it, never raise the budget, never bypass it. (`editstack slots` shows usage.)
7. **Real hardware only when the user allows it, after sim qualification, with a written expected result.**
   Sub-agents never drive hardware.
8. **Do not delete or "clean up" anything you did not create.** Archive instead. Past runs lost a best
   checkpoint and another agent's captures this way.
9. **If the success definition, cameras, sim, or scoring changes, bump `harness_version` in stack.json** (old runs
   then stop counting automatically) and note why in `model/ARCHITECTURE.md` section 5.

---------------------------------------------------------------------------------------------------------------

## 1. Layout

```
stack.json              task, metrics, qualification rules, RAM budget, harness_version, python interpreter
AGENTS.md               this guide (CLAUDE.md points here)
weights/original/       THE model as supplied (read-only)            weights/variants/<v>/  exported variants
weights/best            symlink to the promoted variant (what deployment loads)
model/INVENTORY.md      auto: every tensor, shape, dtype, stats, groups   (editstack decode)
model/SCAN.md           auto: structural-defect hints (padding, untrained slices, dead/tied tensors)
model/ARCHITECTURE.md   VERIFIED description of the forward pass, training loss, inference loop, deployment
model/MODEL_MAP.md      what each part does (probe results, R^2, gains), append-only knowledge register
model/CLAIMS.json       which neurons/channels each variant owns (auto from ctx.claim)
env/adapter.py          load_policy / run_episode (/ episodes / capture): the ONLY sim/model glue
eval/suites/*.json      fixed episode lists: screen, id, ood, heldout (roles: screen|selection|heldout)
edits/<v>/edit.py       PARENT, OWNER, apply(W, ctx); data/ holds fitted arrays
probes/                 shared probe scripts (run with `editstack run probes/x.py`)
agents/<name>/          BRIEF.md (task), REPORT.md (results), scratch scripts and data of one agent
runs/<v>/<suite>/s<k>/  eval results (results.jsonl, per-episode artifacts, DONE)
board/                  LEADERBOARD.md, BEST_HISTORY.md
docs/                   LESSONS.md (evidence), EDIT_COOKBOOK.md (recipes), HARNESS_AUDIT.md (phase-0 checklist)
```

---------------------------------------------------------------------------------------------------------------

## 2. The workflow

Each phase has a gate. Do not start a phase before the previous gate passes. Time boxes are for a model of
~0.5-2 B params with an ~30-60 s sim episode; scale accordingly.

### Phase 0 - Harness audit (lead, <= 30-60 min, before ANY edit)
The single biggest loss in past runs: ~2 days of tuning against sim cameras that did not match the real ones.
Work through `docs/HARNESS_AUDIT.md`. Minimum:
- `env/adapter.py` implemented; `editstack eval original --suite screen --episodes 0-1` runs end to end.
- Sim matches deployment: cameras (pose, FOV, resolution, which stream), calibration, control rate, chunk
  execution, normalisation stats, prompt/task string. Write each check into ARCHITECTURE.md section 5.
- **Success definition agreed with the user and written in stack.json** (it changed twice mid-project last
  time). Use STAGED metrics (e.g. reach / pick / ...) plus an outcome code per failure type (short, long, wide,
  no-attempt...), so separate agents can own separate stages.
- Suites fixed: `screen` (8-12 eps, fast), `id` (in-distribution), `ood` (the generalisation target),
  `heldout` (never used for choosing; run only before promotion). Seeds only change policy noise; placements
  are fixed so comparisons are PAIRED.
- Baseline measured: `original` on id (2 seeds) and ood (3 seeds). Determinism check: same (variant, seed,
  episode) twice gives the same result.
**Gate:** baseline on the board, harness_version set, audit checklist complete.

### Phase 1 - Decode and structural defects (lead alone, <= 1 h, no sub-agents)
The best single win in past runs came here in 74 minutes (8/20 -> 19/20), while a 14-hour interpretability
hunt at the head/neuron level stalled at 5/10.
- `editstack decode`; read INVENTORY.md and SCAN.md.
- Fill ARCHITECTURE.md sections 1-4 FROM THE CODE (model, training loss, inference loop). Look specifically for:
  padded channels; loss masks (`loss[..., :k]`); inputs that are fed at inference but never supervised;
  normalisation stats; sampling loop details (steps, what is fed back); chunk length and how many steps execute.
- For each defect: an exact algebraic fix (section 3, family 1), verified offline, then one screen.
**Gate:** ARCHITECTURE.md sections 1-4 filled with [verified] lines; structural fixes evaluated (kept or rejected
with numbers).

### Phase 2 - Probe: locate the failure (<= 1-2 h)
Before editing behaviour, find WHERE it goes wrong. Cheapest first:
- **Offline single-chunk probes** (seconds each): feed recorded observations (captured during evals) through
  the model, vary one input, measure the output. Use `st.policy(...)` and `adapter.capture` if present.
- **Linear decodability** (`fit.cv_r2`, grouped by episode/placement): is the task variable (e.g. object
  position) linearly present at each layer? Past model: R^2 0.87-0.99 everywhere => perception was fine, the
  read-out was the problem. That one fact redirected all later work to the action side.
- **Gains** (`fit.gain`): per decision step, how much does the output move per unit of the error it should
  correct? Past model: 0.03-0.6 where ~1 was needed (wrist roll 0.03!).
- **Failure taxonomy** from the staged metrics: what fraction of failures are each outcome code? (Past: 49 of 69
  reached-but-not-picked were wrist roll > 25 deg off: one focused fix.)
Record every finding in MODEL_MAP.md (what, where, number, evidence path).
**Gate:** a written causal hypothesis with a number attached ("the policy under-corrects pan by 0.4x beyond
|x|>7 cm; per-chunk gain 0.34").

### Phase 3 - Edit (exactness ladder: always use the highest rung that works)
1. **Exact algebra** (`editstack.ops`): disconnect never-trained inputs (`zero_cols`); exact translations/scalings
   of an input/output axis with compensating biases (`compensate_input_shift`, `rescale_channel`); fold a linear
   map into a layer (`fold_input_map`, e.g. mapping new-camera features back to training-camera features).
   Prove exactness offline: the function-preserving half of the edit must reproduce the original outputs to
   float tolerance before you add the behaviour-changing half.
2. **Closed-form fits** (`editstack.fit`): capture activations X during real rollouts, define the target Y the
   layer *should* produce, solve `fit_delta(X, Y, W, b, lam, rank=1..4)`, save to `edits/<v>/data/`.
   Low rank, grouped CV, and a held-out check of the fit's R^2 before any sim.
3. **Gated circuits** in spare units (`ops.free_units`, `ops.exact_product_pair`, `ctx.claim`): add a new
   mechanism whose output is exactly zero in situations the model already handles (e.g. multiply by
   "planned - current state", which is 0 while holding still; or gate on the out-of-distribution feature).
   This is how generalisation was added without in-distribution cost. Claim every unit.
Never: random search over scales, edits you cannot explain, changing shapes, touching weights/original.
Every edit.py docstring states: hypothesis, evidence path, expected effect (number), exactness argument.
`editstack apply <v>` must show only the tensors you meant to change, with the rank you expect.

### Phase 4 - Evaluation ladder (stop at the first rung that fails)
| rung | what | cost (past model) | pass criterion |
|---|---|---|---|
| a | `editstack apply <v>` diff + offline probe on recorded obs | seconds | expected tensors/rank; target output moves by the predicted amount |
| b | `--suite screen --seeds 0` | ~10 min | no regression on screen, outcome codes move the predicted way |
| c | qualify: `ood` 3 seeds + `id` 2 seeds | 1-2 h | stack.json rules; paired vs parent (board shows wins/losses + sign-test p) |
| d | `heldout` 1 seed | ~20 min | within ~15 pp of selection suites (else overfit) |
| e | real hardware (only if allowed) | user time | the expected result you wrote down beforehand |
Run long evals with `--detach`, then `editstack wait <v> --suite <s>` (one blocking call, no sleep loops).
Two seeds misled twice before (23/24 -> 31/36 at seed 3). Never draw conclusions from a screen.

### Phase 5 - Compose and promote (one owner: the lead)
- Variants form ONE chain via PARENT: each new edit builds on the current best (`edits/<v>/edit.py PARENT=...`).
- After adding each link, re-run rung c; interference appears when two edits correct the same quantity.
- `editstack promote` (requires held-out results) exports, verifies bit-exactness, re-points weights/best and
  logs to board/BEST_HISTORY.md.

---------------------------------------------------------------------------------------------------------------

## 3. Multi-agent orchestration

**What worked:** sub-agents that OWN ONE BENCHMARK STAGE with a numeric target, a fixed suite, an owned folder
and a process budget (approach owned "reach", grasp owned "grip", camfix owned "pass every suite under measured
cameras") produced every best variant. **What did not:** "test this theory" briefs (all three first-wave agents
ended with "keep the baseline"), diagnose-only swarms (27 small read-only agents: 0 accepted candidates), two
agents on one hypothesis, and maintenance agents.

Rules:
- **Phase 0-2 are the lead's job, alone.** Delegate only from Phase 3 on, once the hypothesis has a number.
- **Number of research agents <= RAM budget / (2 x per_process_gb)**, each with <= 2 model processes
  (60 GB / 8 GB -> 3 agents). More agents only lengthen the queue; past agents idled 60-85% waiting for sims.
- **Create each agent with `editstack agent new <name> --stage ... --target ...`** and delegate with
  "Read agents/<name>/BRIEF.md and AGENTS.md, then do it." The brief is the only context it gets.
- **Never give two agents the same hypothesis or the same stage.** Check agents/*/BRIEF.md first.
- **Agents register their own variants** (they appear on the board automatically) and append facts to
  MODEL_MAP.md; the lead merges chains and promotes.
- **The lead does not poll.** Use `editstack wait` / completion notifications, and `editstack status` for a
  one-shot digest. Polling and re-running agents' evals bloated the lead's context to ~780k tokens before.
- **Rendering, videos, GIFs, write-ups and maps go to a separate doc agent** (cheap, high value per minute);
  the lead never spends context on them.
- **No side projects in the same RAM pool** while research agents run (a parallel training project cut eval
  throughput from ~250 to ~60 episodes/h).
- Agents survive network drops: runs are resumable (rerun the same eval command; finished episodes are kept).
  Tell a resumed agent "resume, don't relaunch".

---------------------------------------------------------------------------------------------------------------

## 4. Commands

```
editstack status                         one-shot digest (running evals, RAM, board top, best)
editstack decode | verify                inventory + scan of the original | check it is untouched
editstack edit new <v> --parent <p> --owner <me> --desc "..."
editstack apply <v>                      in-memory apply + diff (tensors, values changed, rank, claims)
editstack eval <v> --suite <s> --seeds 0,1,2 --jobs 2 [--episodes 0-5] [--detach]
editstack wait <v> --suite <s>           block until DONE; prints per-seed counts
editstack board | promote [--dry-run]    leaderboard (writes board/LEADERBOARD.md) | update weights/best
editstack export <v>                     weights/variants/<v> in the original format, verified bit-exact
editstack agent new <name> --stage <s> --target "<numbers>"   brief for a stage-owner sub-agent
editstack claims | slots                 neuron/channel ownership | machine-wide RAM reservations
editstack run probes/x.py [args]         run a probe under the stack's python with the stack importable
```
Python: `from editstack import Stack, ops, fit; st = Stack.find(); a = st.weights("v"); pol = st.policy("v")`.

---------------------------------------------------------------------------------------------------------------

## 5. Anti-patterns (each cost hours before)

- Editing before the harness audit (wrong cameras: ~2 days lost; a stale code copy: undetected for days).
- Interpretability hunts at head/neuron level before checking structural defects and read-out gains.
- Re-planning / horizon / ensembling tweaks as a substitute for fixing the policy (made results worse; reverted).
- Editing during training (training cancels edits; edit the finished checkpoint once).
- Promoting on 2 seeds; selecting dozens of variants on the same placements without a held-out suite.
- Changing the success definition mid-way without bumping harness_version (numbers silently mix).
- `sleep` polling loops; agents waiting for a go-ahead that nobody will give (state the default and proceed).
- Two edits correcting the same quantity (double correction), or sharing a neuron (use claims).
- Large unexplained edits ("214 M values changed to hard-code a controller" was rejected by the user).

## 6. Writing things down

- `model/MODEL_MAP.md`: append facts as `- [date][agent] <fact with number> (evidence: <path>)`.
  Include negative results; they prevent repeats.
- `agents/<name>/REPORT.md`: update as you go (a crash must not lose the result). First line = the result.
- `model/ARCHITECTURE.md`: only verified statements; mark anything else [unverified].
- Edit docstrings: hypothesis, evidence, expected effect, exactness argument.
