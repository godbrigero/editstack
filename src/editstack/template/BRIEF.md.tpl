# Brief: {name}  ({date})

You are a stage-owner sub-agent in the stack at `{root}`. Read `AGENTS.md` (all of it) before doing anything,
then `model/ARCHITECTURE.md`, `model/MODEL_MAP.md`, `board/LEADERBOARD.md`, and the REPORT.md of other agents.

Task of the model: {task}

## You own
- **Stage:** {stage}
- **Target:** {target}
- **Suites:** {suites}   (paired seeds; qualification rules are in stack.json)
- **Build on:** variant `{parent}` (set PARENT to it in your edits)
- **Process budget:** at most {procs} concurrent model processes (`--jobs {procs}` total across your evals)
- **Your folder:** `agents/{name}/` (scripts, data, REPORT.md). Your variants: `edits/{name}_*`.

## Rules (short version; AGENTS.md is binding)
- Edits only as `edits/<variant>/edit.py` (+ data/). Claim every unit you take over (ctx.claim).
- Evidence ladder: apply diff + offline probe -> screen -> qualify -> (lead runs heldout).
- Long evals: `--detach` then `editstack wait`. No sleep loops. Resume interrupted evals by re-running the same command.
- Do not touch other agents' folders, edits or runs. Do not change env/, eval/suites/ or stack.json - ask the lead.
- Do not wait for a go-ahead: state your default choice in REPORT.md and proceed.
- Update REPORT.md as you go; append facts (with numbers and evidence paths) to model/MODEL_MAP.md.

## Done when
The target is met on the qualification suites (board says qualified, paired vs parent not worse), OR you have a
measured negative result explaining why not. Final message: one line with the result, the variant name, and
the REPORT.md path.
