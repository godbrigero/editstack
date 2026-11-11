# Orchestration guide for the lead agent

The lead owns: the harness, the analysis that produces the causal sentence, the chain, promotion, and the user.
Sub-agents own one benchmark stage each. Everything else (renders, write-ups) goes to cheap doc agents.

## When to delegate

- **Phases 0-2: never.** Harness audit, structural fixes and first analysis are serial and context-heavy. Splitting
  them produced 27 agents and 0 accepted candidates, while one lead got 19/20 in 74 min.
- **Phase 3 onwards: yes, once failures split into independent stages** (e.g. "reaches the wrong place" vs
  "reaches but doesn't grip" vs "breaks under the new cameras"). Each stage gets one owner.
- **Not by hypothesis.** "Test theory X" agents (latent, circuit, pancost) all ended with "keep the baseline".
  Hypotheses are tools a stage owner picks from, not jobs.

## How many

`research agents <= ram_budget_gb / (2 * per_process_gb)`, at 2 model processes each. With 60 GB and 8-9 GB per
process that is 3. Past agents idled 60-85% of the time waiting for sims, and adding agents only lengthened the
queue. Nothing else that loads models may share the RAM pool (a side training project cut throughput 4x).

## Briefing (the brief is the agent's only context)

`editstack agent new <name> --stage <stage> --target "<numbers>" --parent <variant>` writes agents/<name>/BRIEF.md.
Then add to it:
- the causal sentence for this stage and its evidence paths;
- what has been tried already (MODEL_MAP negative results) and must not be repeated;
- the default to take when a choice comes up ("don't wait for a go-ahead": two agents once idled ~80 min).
Delegate with: "Read agents/<name>/BRIEF.md and AGENTS.md, then do it."

## While they run

- Don't poll. Wait for completion notices; use `editstack status` for a one-shot digest. ~75 result checks and 27
  sleeps grew the lead's context to ~780k tokens before.
- Don't re-run an agent's evals yourself. Read its run dirs and the board.
- Watch `model/CLAIMS.json` and MODEL_MAP.md for overlaps between agents. Resolve them early.
- Network drops: tell the agent "resume, don't relaunch". Evals resume on re-run.

## Merging and promotion

- One chain. When two stage owners both qualify, rebase one onto the other (PARENT), re-qualify, and check for
  double correction.
- Promote only via `editstack promote` (held-out required). Record the reasoning in board/BEST_HISTORY.md.
- Real hardware: only if stack.json allows it, after promotion, with a written expected result. The lead runs it,
  never a sub-agent.

## Doc agents

Give each one a narrow write scope (one folder) and no sim access unless it is rendering an existing run. These
were the highest value per minute (4-51 min each). They keep rendering out of the lead's context.
