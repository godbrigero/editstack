# Agent guide: editing this model's weights

> **The evidence in these guides comes from a PAST reference project** (a SmolVLA policy on an SO-101 arm, Nov 2026).
> Its numbers, placements, tensor names and neuron indices describe THAT model, not the one in this stack. Use them
> as lessons about method only. Facts about YOUR model come only from `model/` and your own runs in `runs/`.

You are editing the weights of ONE model (this stack) by hand: no gradient training and no new data, unless the user
explicitly says otherwise. This guide and `docs/` distil ~5 days and ~85 agent sessions of hand-editing a SmolVLA
robot policy, corrected by an independent evidence audit. Every rule has evidence in `docs/LESSONS.md`. The
guidance is fast because it removes guesswork: every edit is derived, every claim is measured, and every number is
comparable. It is a living knowledge base: Phase 6 is how you improve it.

**Start:** run `editstack status`, then read `stack.json` (task, metrics, qualification rules), `model/ARCHITECTURE.md`,
`model/MODEL_MAP.md` and `board/LEADERBOARD.md`. Then read the guides you need (`docs/README.md` is the index):

| situation | read |
|---|---|
| first time in this stack | this file, then `docs/HARNESS_AUDIT.md` |
| understanding the model / locating a failure | `docs/ANALYSIS_GUIDE.md` |
| only numbers + a description (no code, anonymous tensors) | `docs/NUMBERS_ONLY_EDITING.md` |
| choosing what to do about a measured symptom | `docs/strategies/INDEX.md` (30 strategies + selection flow) |
| executing an edit | the strategy file + `docs/EDIT_COOKBOOK.md` |
| you are the lead, or you are about to start agents | `docs/AGENT_FLEET.md` (its first table is the quick reference; §3 below is the summary) |
| a number or rule seems wrong | `docs/LESSONS.md` (single source of numbers), then Phase 6 |

**Scale the process to the model.** The full process below is for slow, expensive evals (seconds to minutes per
episode, real hardware, shared RAM). If an episode takes about a second and there is no hardware or camera: skip the
real-frame and RAM steps, run evals in the foreground (no `--detach`/`wait`), and fill ARCHITECTURE.md only with the
facts your edit relies on (a few lines). A solo agent can skip AGENT_FLEET.md, and NUMBERS_ONLY_EDITING.md is only for
models without code. The rules in section 0 and the evaluation ladder still apply. A blind trial on a toy stack fixed the
model in 16 tool calls by going straight from SCAN.md to S01 to `qualify` to `promote`.

---------------------------------------------------------------------------------------------------------------

## 0. Ground rules (non-negotiable)

1. **`weights/original/` is read-only.** Every variant is `original + edit chain`, applied in memory. Every eval
   checks the original's integrity automatically; `editstack verify` does a full hash check.
2. **Every edit is code in `edits/<variant>/edit.py`** (+ fitted arrays in `edits/<variant>/data/`). Shared exact
   helpers live in `env/` (they are part of every variant's fingerprint). Nothing is edited in notebooks or by hand.
3. **Scripts live in exactly three places:** `env/` (adapter + helpers, owned by the lead), `probes/` (shared probes),
   and `agents/<name>/` (one agent's scratch). Nothing at the stack root.
4. **Never state a result you did not measure.** Every number cites a run dir, a probe output file or a command.
5. **Only the leaderboard decides.** A variant is better only if `editstack board` says so under the qualification
   rules: complete seeds only, paired against its parent, then held-out before promotion.
6. **RAM: every model process goes through editstack** (eval, `Stack.policy`), which takes a machine-wide slot
   (`~/.config/editstack/machine.json`). Never load a model outside it.
7. **Real hardware:** only when stack.json allows it, after promotion, with a written expected result. Sub-agents
   never drive hardware.
8. **Never delete or "clean up" what you did not create.** Stale runs are archived automatically.
9. **Changed the sim, cameras or adapter?** Bump `harness_version` and old runs stop counting. **Changed only the success
   definition?** Bump `metric_version` and run `editstack rescore` (no sim). Note why in ARCHITECTURE.md section 5.
10. **Read any scorer before trusting it.** A past "scorer" only counted its own invocations.

---------------------------------------------------------------------------------------------------------------

## 1. Layout

```
stack.json              task, metrics (+metric_version), qualification rules, harness_version, python, deploy_links
AGENTS.md               this guide                          docs/        the knowledge base (index: docs/README.md)
weights/original/       THE model as supplied (read-only)   weights/variants/<v>/  exports   weights/best -> promoted
model/INVENTORY.md      auto: tensors, shapes, dtypes, stats, groups (editstack decode)
model/SCAN.md           auto: structural-defect hints (padding, init-like slices, dead/tied tensors, I/O dims)
model/ARCHITECTURE.md   VERIFIED forward pass, training loss + masks, inference loop, deployment, sim-vs-real
model/MODEL_MAP.md      append-only register of measured facts about this model (with evidence paths)
model/CLAIMS.json       units each variant owns (from ctx.claim; changing someone else's claimed slice is an error)
env/adapter.py          load_policy / run_episode (+ episodes, capture, rescore): the only sim/model glue
eval/suites/*.json      fixed episode lists; roles: screen | selection | heldout
edits/<v>/edit.py       PARENT, OWNER, apply(W, ctx)          probes/   shared probes   agents/<name>/  per-agent
runs/<v>/<suite>/s<k>/  results.jsonl, per-episode artifacts (trajectories, observations), DONE
board/                  LEADERBOARD.md, BEST_HISTORY.md, knowledge patches
```

---------------------------------------------------------------------------------------------------------------

## 2. The workflow

The gates are what must be true before the next kind of ACTION, not before reading or thinking. Time boxes assume
a ~0.5-2 B parameter model and a ~30-60 s sim episode.

### Phase 0: Harness (lead, about 30-60 min of attention; the sim time runs in the background)
- Make the adapter run: `editstack eval original --suite screen --episodes 0-1`.
- **Ask the user now** for one real frame per camera and the deployment details (control rate, chunk execution,
  calibration), then compare them with the sim. A camera mismatch found on day 4 forced a whole camera-transfer
  stage. Work through `docs/HARNESS_AUDIT.md`.
- Agree the success definition with the user. Use STAGED metrics (e.g. reach / pick) plus outcome codes, so effects
  are visible per stage. A first wave once looked useless only because it was scored on PICK alone.
- Fix the suites: `screen` (smoke only), `id`, `ood` (the target), and `heldout` (never used for choosing).
- **Start the baselines detached now** (`original` on id with 2 seeds and ood with 3 seeds, `--detach`). Don't wait
  for them; go to Phase 1.
- **Gate (before the first EDIT is evaluated):** the adapter is deterministic (rerun one episode with `--fresh` and
  compare results.jsonl: identical); the sim matches deployment on the checked items; baselines are on the board.
  Rule thresholds count episodes ACROSS complete seeds (2 seeds x 20 episodes = 40 >= min_episodes 30).

### Phase 1: Structure first (lead alone, <= 1 h, no sub-agents)
- Run `editstack decode`, then read SCAN.md. Read the model, training and inference code, in the order given in
  `ANALYSIS_GUIDE.md` step 1. With numbers only, follow `NUMBERS_ONLY_EDITING.md`.
- Hunt for structural defects: padded/unsupervised channels, loss masks, train-vs-inference input mismatches,
  normalisation stats, the sampling loop, chunk execution.
- Fix each defect with an exact strategy (S01-S03 in `docs/strategies/`), verify offline, then evaluate.
- **Gate:** the facts your edits rely on are in ARCHITECTURE.md marked [verified]; you don't need to fill
  everything.

### Phase 2: Locate the failure (<= 1-2 h; offline probes, seconds each)
- Use `ANALYSIS_GUIDE.md` steps 2-3: failure taxonomy from outcome codes, signed errors vs the task variable,
  single-chunk probes on saved observations (`adapter.capture`), linear decodability (`fit.cv_r2`, grouped),
  gains (`fit.gain`), source attribution, path ablation.
- **Gate:** a causal sentence with a number ("pan gain 0.34 beyond |x| > 7 cm; front camera alone aims the first
  move"), plus a predicted effect of the fix.

### Phase 3: Edit (choose from the strategy library)
- Open `docs/strategies/INDEX.md` and follow its selection flow from your symptom. Prefer the highest rung that
  works: **exact algebra > closed-form fit > gated circuit**. Read the "failed" strategies (S17-S24) before trying
  anything that resembles them.
- Each edit.py docstring states: hypothesis, evidence path, expected effect (a number), and an exactness argument.
- `editstack apply <v>` must show only the intended tensors, at the rank you expect, with your claims. Reserve spare
  units the moment you choose them (`editstack claims reserve`), so a parallel agent can't take them.

### Phase 4: Evaluation ladder (stop at the first rung that fails)
`editstack qualify <v>` runs rungs b-d in order and stops at the first failure. The table says what each rung means.
| rung | what | pass criterion |
|---|---|---|
| a | `editstack apply` diff + offline probe on saved observations | the expected tensors and rank; the target output moves by the predicted amount |
| b | target suite, seed 0 (`--detach`, then `editstack wait`) | **paired early stop**: the board's "vs parent" must not show more losses than wins. If it does, stop and record it |
| c | qualify: target suite 3 seeds + id 2 seeds | stack.json rules (complete seeds only); paired vs parent |
| d | `heldout`, 1 complete seed | gap to the selection rate <= `max_heldout_gap`; promotion is blocked otherwise |
| e | real hardware (only if allowed) | the expected result you wrote down beforehand |
- `screen` is only a smoke test. 8 episodes cannot show a regression.
- Two-seed results misled three times. Never promote on fewer than the rules require.

### Phase 5: Compose and promote (one owner: the lead)
- Keep one chain via PARENT. Before adding a link, ask: does an earlier link already correct this quantity?
  Double correction was the main source of interference. Re-qualify after every link.
- `editstack ablate <v> --suite <target> --seeds 0,1,2` (leave-one-out) before promoting. In one past project,
  pad-zero plus one servo got the whole gain; four later circuits added nothing. Drop links that don't pay.
- `editstack promote` exports with bit-exact verification, re-points `weights/best` and any `deploy_links`, and
  logs the change to board/BEST_HISTORY.md.

### Phase 6: Record and evolve the knowledge (every agent, every experiment, a few seconds)
- After EVERY experiment, positive or negative:
  `editstack learn "<what happened, with numbers>" --outcome win|loss|neutral|mixed|info --evidence <run dir> --strategy Sxx`.
  This appends to `docs/LEARNINGS.md` and to that strategy's evidence log.
- When the evidence changes a strategy's status (e.g. untested → partial → proven, or proven → failed on this model
  class), edit the `Status:` line following `docs/strategies/HOW_TO_UPDATE.md`. A new technique gets the next
  free S-number.
- Facts about THIS model go to `model/MODEL_MAP.md`. Lessons about editing IN GENERAL go to LEARNINGS.md, and then
  into the guide they change.
- `editstack docs contribute` writes a patch of these improvements for the central editstack repo. Opening a PR
  publishes it, so ask the user first. `editstack docs sync` pulls improved guides into this stack and never
  overwrites your local edits.

---------------------------------------------------------------------------------------------------------------

## 3. Multi-agent work (full guide: docs/AGENT_FLEET.md)

- **Phases 0-2: one agent** holding the whole context. Delegate from Phase 3 once there is a causal sentence.
- **Split by measurable quantity:** a stage (reach / grip / cameras) or a mechanism. Every agent gets a staged metric
  that can SEE its effect, a numeric target, a fixed suite and its own folder. Never give two agents the same
  quantity.
- **How many:** `N = floor(min(P_ram, P_sat) / 2)` research agents at 2 processes each. That is 2 on a 60 GB machine
  with ~9 GB processes, 3 at most. Agents idled 25-86% waiting for sims; more agents only lengthen the queue.
- **Tiers:**

  | role | tier |
  |---|---|
  | lead, stage owners, reviewers | frontier |
  | doc / render | mid |
  | jobs a checker can verify mechanically | fast tier is fine |
  | interpreting or designing | never the fast tier |
- **Brief** with `editstack agent new`, plus the causal sentence, what was already tried, and the default for each
  choice. Agents state defaults and proceed; the lead never waits on a go-ahead either.
- **The lead does not poll.** Use completion notices, `editstack wait` and `editstack status`. Rendering, videos
  and write-ups go to a separate doc agent.
- **Independent review** before promoting or publishing numbers: a fresh agent given the code and the evidence, not
  your conclusions.

## 4. Commands

```
editstack status                                   digest: running evals, RAM, board top, best
editstack decode | verify                          inventory + scan | full integrity check of the original
editstack edit new <v> --parent <p> --owner <me>   then: editstack apply <v>   (diff, rank, claims; registers nothing)
editstack eval <v> --suite <s> --seeds 0,1,2 --jobs 2 [--episodes 0-5] [--detach]
editstack wait <v> --suite <s>                     blocks until that job ends; prints per-seed results
editstack qualify <v>                              ladder: target seed 0 -> paired early stop -> qualify seeds -> held-out
editstack board | ablate <v> --suite <s> | promote [--dry-run]
editstack rescore [<v>]                            recompute metrics from stored trajectories after metric_version++
editstack export <v>                               weights/variants/<v>, original format, verified bit-exact
editstack agent new <name> --stage <s> --target "<numbers>"
editstack learn "<text>" --outcome win|loss|... --evidence <path> [--strategy Sxx]
editstack docs sync | contribute | index           evolve the knowledge base
editstack claims [reserve <v> <tensor> <axis> <idx>]   list / reserve units the moment you pick them
editstack slots | run probes/x.py
```
Python: `from editstack import Stack, ops, fit`; `st = Stack.find()`.
- `a = st.weights("v")` returns an Applied object: `a.tensors` (dict name -> array, the full weights), `a.changed`
  (names), `a.stats` (per-tensor diff stats) and `a.chain`.
- `pol = st.policy("v")` takes a RAM slot and builds the runnable model through the adapter.
- `st.adapter().capture(pol, obs, names)` records activations, if the adapter has it.
In edit.py, `PARENT = None` means the variant builds directly on the original model.

## 5. Anti-patterns (each cost hours before; evidence in LESSONS.md and strategies S17-S24)

- Editing before the harness check, especially cameras. Ask for real frames on day 1.
- Starting with head/neuron interpretability before structural defects and read-out gains.
- Re-plan/horizon tweaks instead of fixing the policy. Editing during training. Global output gains. Naive bias
  nudges. Hard-coding a controller.
- Promoting on 2 seeds. Choosing among dozens of variants on the same placements without a held-out suite.
- Scoring agents on a metric that can't see their effect.
- Two agents on one quantity. Two edits correcting the same quantity. Shared neurons (CLAIMS.json catches these).
- Polling with sleep loops. Waiting for a go-ahead. Side projects in the same RAM pool.
- Unexplained edits ("214 M values changed to hard-code a controller" was rejected by the user).
