# Brief: {name}  ({date})

You are a sub-agent that owns one measurable quantity in the stack at `{root}`. Read `AGENTS.md` (all of it) before doing anything,
then `model/ARCHITECTURE.md`, `model/MODEL_MAP.md`, `board/LEADERBOARD.md`, and the REPORT.md of other agents.

Task of the model: {task}

## You own
- **Stage:** {stage}
- **Target:** {target}
- **Suites:** {suites}   (paired seeds; qualification rules are in stack.json)
- **Build on:** variant `{parent}` (set PARENT to it in your edits)
- **Process budget:** at most {procs} concurrent model processes (`--jobs {procs}` total across your evals)
- **Your folder:** `agents/{name}/` (scripts, data, REPORT.md). Your variants: `edits/{name}_*`.

## Context you must fill in before delegating (lead)
- Causal sentence for this stage, with numbers and evidence paths:
- Already tried (do not repeat; see MODEL_MAP.md negative results and docs/strategies evidence logs):
- Candidate strategies from docs/strategies/INDEX.md:
- Defaults to take without asking:

## Rules (short version; AGENTS.md and docs/AGENT_FLEET.md are binding)
- Your metric must be able to SEE your effect (a staged metric for your stage). If it can't, say so first.
- Edits only as `edits/<variant>/edit.py` (+ data/). Claim every unit you take over (ctx.claim).
- Evidence ladder: apply diff + offline probe -> target suite seed 0 + paired early stop -> qualify -> (lead runs heldout).
  `screen` is a smoke test only.
- Long evals: `--detach` then `editstack wait`. No sleep loops. Resume interrupted evals by re-running the same command.
- Do not touch other agents' folders, edits or runs. Do not change env/, eval/suites/ or stack.json - ask the lead.
- Do not wait for a go-ahead: state your default choice in REPORT.md and proceed.
- Evaluate in this order: seed 0 on the target suite, then the paired early stop (board "vs parent"), then qualify.
- Update REPORT.md as you go; append facts (with numbers and evidence paths) to model/MODEL_MAP.md.
- After every experiment: `editstack learn "<result with numbers>" --outcome ... --evidence ... --strategy Sxx`.

## Done when
The target is met on the qualification suites (board says qualified, paired vs parent not worse), OR you have a
measured negative result explaining why not. Final message: one line with the result, the variant name, and
the REPORT.md path.
