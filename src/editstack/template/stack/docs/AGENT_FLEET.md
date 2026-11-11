# Agent fleet: who to start, on which model, with what brief

Read this if you are the lead and are about to start, brief, wait on, resume or stop another agent. `../AGENTS.md` §3
is the short version. The roster table below is the quick reference; the rest of this file is the reasons, the numbers,
the brief templates (§7) and the procedure. Evidence: about 85 Claude sessions and one Codex thread from the SmolVLA SO-101 project (Nov 5–10 2026).
Times are UTC. A session id such as `e5bc0951` refers to a Claude Code transcript. E1–E10 below refer to sections of the retrospective's evidence log (not shipped).

Where the evidence is thin, a recommendation is marked **[judgement]**. Everything else has a number behind it.

---------------------------------------------------------------------------------------------------------------

## 0. Summary

| role | how many | tier | when |
|---|---|---|---|
| Lead / orchestrator | 1, for the whole task | frontier (Opus) | always. Alone for Phases 0–2 |
| Stage owner (research) | ≤ `N_research` (§3), 2 on a 60 GB machine, 3 at most | frontier (Opus), high effort | from Phase 3 on, one per independent stage |
| Doc / render | 1 at a time, one folder each | mid (Sonnet). Frontier if the doc needs analysis | whenever the user wants pictures, pages or write-ups |
| Status digest | 0 agents. Use `editstack status` | none (or fast, if you must) | instead of polling |
| Independent reviewer / verifier | 1 per claim set, before promotion or publication | frontier | before you promote, publish numbers or write rules |
| Read-only analyst (retrospective) | 3–5 in parallel, one source each | frontier (mid for pure extraction) | after the task, or to digest large logs |
| Cheap parallel explorer (fast tier) | 0 by default. ≤ 3 when the job is mechanical and checkable | fast (Haiku) | only for fully specified, checkable jobs, always reviewed |

The five rules that matter most:
1. **Give every agent a staged metric that can see its effect, a numeric target and a fixed suite.** Without these,
   good work looks like a null (§1.2).
2. **Never give two agents the same hypothesis or stage** (latent and circuit both fixed reach, in parallel, §1.2).
3. **Don't wait for a go-ahead you don't need. State the default and proceed.** This applies to the lead too: two
   ready agents sat unlaunched for 77 min (§5.6).
4. **Respect the shared RAM pool.** Use slots and at most 2 model processes per agent. Run nothing else that loads
   models. Detach long runs and give them DONE markers (§3, §5.4).
5. **Write REPORT.md as you go, and finish with a fixed-format message.** Network drops killed turns in 4 agents. The
   files and the detached sims are what survived (§5.5).

---------------------------------------------------------------------------------------------------------------

## 1. Roles, with measured outcomes and tier

Every Claude agent in the main project ran on Opus (`message.model = claude-opus-5-5`), including all doc agents (E1).
The only other tiers that ran are:
- 27 Haiku children under a Codex (GPT-6) parent;
- 2 Sonnet doc agents in the hybrid project.

So "Opus did X" is well measured, Sonnet has n = 2, and Haiku has n = 27, but under heavy confounds.

### 1.1 Lead / orchestrator
- **Owns:** the harness, Phases 0–2, the causal sentence, the chain, merges, promotion, real hardware and the user.
- **Measured:** b078391b ran 11-05 22:40 → 11-10 19:39 (117 h span), with only **557 min active** (gaps < 5 min).
  - 746 tool calls, 19 delegations and 26 follow-ups.
  - Alone, it derived pad0 (8/20 → 17/20 by 23:27, 47 min after start; deep8 19/20 at 23:54 on one seed) and pan ×1.7 (`scale_axis`, verified offline
    before any sim, 11-07 01:24). pan ×1.7 was the base of every later best.
  - It built the shared tooling the agents depended on: `bench.py` staged metric, `slots.py`, `CONTEXT.md`,
    `update_best.py`.
- **Cost:** the lead read **317 M cache tokens**, about as many as all 9 research agents combined (330 M).
  - Context peaked at **780,301 tokens (11-07 21:30)**, while it also ran the hybrid side project.
  - On the research day the median was 512k.
  - A long-lived lead re-reads its whole context on every turn, so lead context is the largest token cost in the fleet.
- **Tier: frontier.** The lead holds the most context and makes the decisions that are hardest to undo (promotion,
  merges, hardware). Both successful orchestrators here were Opus. The Codex/GPT-6 parent also had frontier capability
  but stalled at 5/10 for ~14 h, so tier is necessary, not sufficient.

### 1.2 Stage owner (research sub-agent)

| agent | brief type | wall / active min | tools | outcome | in best lineage |
|---|---|---|---|---|---|
| latent | mechanism ("theory to test"), PICK-only metric | 114 / 50 | 79 | radial error −3 cm → ~0. Read as a "null" on PICK, then 23/24 REACH once REACH existed | yes, via r1a_xg |
| circuit | mechanism, PICK-only | 121 / 52 | 76 | no pick gain. Its neuron-pair construction became the template for every later circuit | method only |
| pancost | measure a cost | 143 / 37 | 98 | proved and explained pan17's in-distribution cost (73 vs 63/80, 1.5× over-aim), which became approach's spec | spec only |
| servo | mechanism ("your theory to test") | 214 / 82 | 153 | **sp6 = first best-thus-far** | yes |
| approach | stage owner (REACH), qualify r1a first | 346 / 90 | 179 | p17gs, then **p17gshr** (near 35/36 R, ID 40/40 R, 34/40 P) | yes |
| grasp | stage owner (R→P), own hover harness | 350 / 49 | 157 | roll diagnosis (49/69 R = misalignment > 25°). grh20 best for 2 h. Its roll link is within noise | yes (link neutral) |
| shadow | diagnostic + "remove losslessly" | 539 / 96 | 237 | key real-world finding (shadows off: P 32 → 26). Its edit double-corrects on top of p17gshr | no |
| camfix | fixed suite + "iterate until every case passes" | 238 / 179 | 229 | found a stale `repo/` camera copy. **cf2** under measured cameras: ID 58/60 R. cf3p25: 59/60 R, 51/60 P | yes |
| seek | reproduce a real failure, then fix | 200 / 72 | 131 | sk5p REACH 10/24 → 22/24 on its reproduction set. Not qualified | no |

- **Success rate:** 5/9 had an edit that became best-thus-far. 2 more (circuit, pancost) supplied the method or spec of
  one. 2/9 (shadow, seek) produced findings only.
- **What predicted success:** not "stage vs hypothesis".
  - Servo had a hypothesis brief and produced the first best.
  - Wave 1 (latent, circuit, pancost) was judged on PICK alone, a metric that could not see reach fixes.
  - What mattered:
    - a **metric that sees the agent's effect** (REACH/PICK staged labels from 11-07 03:46; grasp's hover harness;
      seek's reproduction set);
    - a **numeric target**;
    - **pointers to prior artefacts** (approach: "Qualify the reach fix (latent r1a)", "circuit's exact, drift-free
      neuron-pair construction");
    - **one owner per quantity**.
  - Latent and circuit both attacked reach, in parallel, and both reported "reach fixed, picks not better". One of them
    was wasted.
- **Time profile:**
  - 25–86% of wall time sat in gaps > 5 min, waiting on sims. 7 of 9 agents were above 55%.
  - Camfix (25%) was the exception: it kept a queue of short jobs running and reviewed results as they landed.
  - Sleep commands per agent ranged from 0 to 95 (shadow).
- **Tier: frontier, high effort.** All 9 were Opus. There is no measured run of a smaller model as stage owner.
  - Tokens are not the bottleneck: a stage owner averaged **142k output tokens over 4.2 h**, and wall time is set by
    the sim.
  - Mistakes are what cost time. Each wasted closed-loop check costs 1–2 h of wall under the RAM cap. The Haiku
    reviews show exactly that kind of mistake: sign errors, wrong axes, wrong units.
  - **[judgement]** Never downgrade this role to save tokens. The savings are a few percent of the fleet's tokens, and
    one sign error costs hours.

### 1.3 Doc / render agent
- **Measured:**
  - 9 Opus doc agents in the main project, 4–51 min each (167 min total), 21–74 tools, 0.38 M output tokens in total.
    All 9 delivered.
  - 3ccd9fef and 18b03fd0 re-renders matched the research runs bit for bit, which doubled as a free determinism check.
  - ee5fcf7b (netmap) re-analysed latent's grid and found r − ŷ linearly decodable. The lead forwarded that as a lead.
  - 16e944b2 backfilled 30 MODEL_MAP entries in 6 min.
  - 2 **Sonnet** doc agents (hybrid) delivered in 8 and 11 min. 4c2c961b recomputed outcomes from fresh renders and
    found that they "match the saved suite scores".
- **Failure seen:** two overlapping doc briefs launched 4 min apart (11112160 at 01:24, ee2243b0 at 01:28). The second
  reported "Most of this already existed from the earlier pass".
- **Why it pays:** the user asked for documentation to go to a separate agent so the lead would not lose context
  (11-06 23:52). GIF/HTML work never entered the lead's context.
- **Tier: mid (Sonnet) for renders, pages and GIFs** (n = 2 Sonnet, both successful, and the job is mechanical).
  - Use **frontier** when the doc needs analysis, e.g. a network map, or a MODEL_MAP backfill that must judge what is a
    fact.
  - **[judgement]** Fast tier only for pure file conversion.

### 1.4 Status / digest
- **Measured:** no dedicated agent existed. Instead the lead ran 77 Bash checks on agent folders, 24 `slots.py`, 33
  `bench.py` and 27 `sleep` commands, and answered "how's the progress for each of the 4 agents?" (11-07 02:08) by
  reading partial run dirs into its own context.
- **Recommendation:** don't start an agent for this. Run `editstack status` once (one-shot digest: running evals, RAM,
  board top, best) and read the board.
  - **[judgement]** If a digest needs judgement across many REPORTs, use a one-shot fast-tier or mid-tier agent with a
    read-only brief and a 15-line output limit. That keeps the reading out of the lead's context.

### 1.5 Independent reviewer / verifier
- **Measured (this retrospective):** one Opus reviewer ran 638 s, 48 tools and 209k tokens. Its findings:
  - the headline rule ("stage owners beat hypothesis testers") was wrong as stated;
  - a table number was wrong (cf2 58/60, not 59/60);
  - a mechanism was claimed that never existed (a "claim list");
  - the two-seed misfires were undercounted (3, not 2);
  - a comparison was confounded (74 min vs 14 h).
- **Earlier:** the Codex parent's review of the Haiku children put corrections on **14 of 18** tracks, including a
  reversed sign (`haiku/parent_review.json`).
- **Tier: frontier.** Verification has to be at least as strong as the work it checks.
  - Give it the artefacts and the claims, not your conclusions.
  - Run it before promotion, before publishing numbers, and before turning lessons into rules.

### 1.6 Read-only analyst (retrospective / log digestion)
- **Measured:** 4 Opus analysts launched in one message (01:55:30–01:56:04).
  - Each ran 206–416 s, with 18–47 tools and 98k–235k tokens.
  - All four were done 7 min after launch.
  - Their notes were mostly right, but held the over-readings above. The reviewer caught them.
- **Tier: frontier, with mandatory review.** **[judgement]** Mid tier is fine for pure extraction ("count X per
  session, give a table"). It is not fine for "what does this mean".

### 1.7 Cheap parallel explorer (Haiku swarm)
- **Measured:** 27 Haiku sessions (11-05 17:02–23:34).
  - Mean 8.6 min and 35 tools each; 233 min total; 2.55 M output tokens. That is twice the output of all 9 Opus
    research agents.
  - **18 tracks reviewed, 0 candidates accepted, 14 corrected.**
  - Errors: a reversed flow-to-action sign; a "range" probe along a nearly vertical axis with cubes floating or
    intersecting the table; control units labelled as degrees; force summed over both sides reported as per-side; a
    residual decomposition presented as a causal ablation.
  - The parent's best stayed at 5/10 from 11-05 10:04 to the end.
- **What the evidence does *not* say:** that Haiku is useless.
  - Zero accepted was guaranteed by design: the briefs forbade editing weights, scoring, fitting and full suites.
  - The parent was stuck and kept working in parallel (7.1 of 7.4 h active).
  - The briefs were 3–6k characters of fences around one prescribed computation, with spaces stripped next to digits
    ("still5/10 with88332", "heads10/11", "factor2").
  - Tier, brief and parent are not separable in this data.
- **What Haiku did well:**
  - bit-exact reproduction gates ("All six baseline replays are bit-exact", "reproduction checks pass (34/34)");
  - construction and arithmetic audits that passed ("every check passed", 8ca746c3);
  - staying inside its filesystem fences (no protected-folder violation in 18 reviews);
  - ~9-min turnaround.
- **What it did badly:** interpretation. Signs, units, axes and causal claims.
- **Tier recommendation [judgement]:** use the fast tier only for jobs whose output can be checked mechanically:
  reproduce a run bit-exactly, audit arithmetic against a formula, extract numbers from logs into a table, render
  existing trajectories, inventory files. Never for probe design, interpretation or edit design.
  - Run at most 3 at a time, each with a different job.
  - A frontier agent reviews every result before it is used.
  - A stage owner that is allowed fast-tier children rarely uses them: shadow was explicitly allowed Haiku children and
    started none in 539 min. Its jobs were sim-bound, and the brief capped it at "2 model processes for you AND your
    sub-agents combined".

---------------------------------------------------------------------------------------------------------------

## 2. Model tiers and provider mapping

| tier | Anthropic | OpenAI / Codex | Google | use for |
|---|---|---|---|---|
| frontier | Opus (this project: `claude-opus-5-5`, effort high) | the flagship GPT model Codex runs by default (this project: GPT-6), high reasoning effort | Gemini Pro-class | lead, stage owner, reviewer, analyst, analysis-heavy docs |
| mid | Sonnet (`claude-sonnet-5-5`) | the standard non-flagship GPT / Codex model | Gemini Pro-class at lower effort, or Flash at high effort | renders, pages, GIFs, extraction-only analysts |
| fast | Haiku (`claude-haiku-5-5`) | mini-class GPT | Flash / Flash-Lite-class | mechanical, checkable jobs only (§1.7) |

- The non-Anthropic rows are **[judgement]** by capability class. Only GPT-6 (as an orchestrator) ran here. Check
  current model names before using the table.
- Effort/reasoning settings matter as much as the tier.
  - The hybrid orchestrator set `effort: high` on every child, and every child delivered something.
  - The Codex parent ran Haiku at `effort: medium`.
- **Name the model explicitly in every delegation.** The lead set a target on only 2 of 19 delegations, and the other
  17 silently got the default. Write down which model each agent ran on, e.g. in the agent's REPORT.md header.

---------------------------------------------------------------------------------------------------------------

## 3. How many agents

### 3.1 The binding constraint is the sim, not tokens
- Real footprint is ~8 GB per SmolVLA process (peak ~8.7 GB; `ps` RSS shows 1–4 GB). 15+ processes hit ~110 GB
  (11-07 03:17). `slots.py` then admitted a process only if measured total + 9 GB ≤ cap, with MAX_PROCS = 6 under a
  60 GB cap.
- Throughput vs busy processes (4,289 episodes, from trajectory mtimes, E6):

| busy model processes | episodes/h | s per episode |
|---|---|---|
| ~1 | ~55–120 | 33–34 (uncontended) |
| ~2 | ~125 (max 200) | ~40 |
| ~4 | 225–290 | 47–56 |
| ~4.7 | **326 (peak)** | 52 |
| ~6 | 254 | 83 (pre-slots, RAM-thrashing) |
| ~4 + a training project in the same pool | 65–69 | — |

- **Throughput saturates at about 4–5 concurrent processes on this machine**, before the RAM cap's 6. Past that, each
  episode slows and the total falls.
- One process doing anything else heavy (the hybrid training) cut throughput about 4x.

### 3.2 Formula
```
P_ram   = floor((RAM_cap_GB - headroom_GB) / footprint_GB)    # 60 GB: floor((60-9)/8.7) = 5 (slots allowed 6)
P_sat   = processes at which episodes/h stops rising           # measure once: run 1, 2, 4, 6 procs for 20 min each
P_max   = min(P_ram, P_sat)                                    # here 4-5
N_research = max(1, floor(P_max / p_agent))                    # p_agent = 2 model processes per stage owner -> 2
           (+1 only if one agent is in a long offline phase, e.g. designing or fitting, and will not hold its slots)
N_doc      = 1 at a time if it renders (1-2 procs, counted in P), or any number if it does not load models
N_fast     = 0 by default; <= 3, CPU-only, no model processes
```
- With 60 GB and SmolVLA, this gives **2 stage owners, 3 at most**. Wave 2 ran 3 (approach, grasp, shadow) and fit
  until the hybrid project took the pool. Wave 1 ran 4 plus renderers and caused the 110 GB incident.
- Agents are not processes. A stage owner holds its slots while it waits on sims (25–86% idle; 7 of 9 above 55%), so a third agent only
  helps if the first two leave slots free. Check `editstack slots`. If slots are full for > 30 min, an extra agent only
  lengthens the queue.
- **When adding agents stops helping:**
  - all slots are busy;
  - two agents would share a stage or a quantity;
  - or the new agent's metric can't see its effect yet. Build the metric first.

---------------------------------------------------------------------------------------------------------------

## 4. Decision procedure

Work through the steps in order. Each says who runs and on which tier.

**Step 0. Phases 0–2 (harness audit, decode, first causal sentence). Lead alone, frontier.**
- Start the baseline evals detached while you read code and run offline probes. Gate edits on the baseline, not your
  reading.
- Allowed help:
  - one doc agent (mid tier) if the user wants pictures;
  - fast-tier jobs only for mechanical checks you would otherwise do by hand (inventory, bit-exact reproduction).
- Why alone: every early structural win came from one agent holding the whole context. pad0 took 74 min, scale_axis was
  verified offline before any sim, and the staged benchmark came from the lead. The swarm that ran during this phase
  produced no edits, partly by design (§1.7).
- Exit when you have a causal sentence with a number, a staged metric that sees each failure stage, and a fixed suite.

**Step 1. Failures split into independent stages? Start wave 1 of stage owners (frontier, high effort).**
- One agent per stage that the staged metric isolates (e.g. REACH, P given R, camera transfer). At most `N_research`.
- If there is only one stage, keep it yourself, or give it to one owner while you build the next harness.
- **Mechanism briefs are fine** if the metric can see the mechanism's effect (servo → sp6). Never two agents on one
  quantity. Check `agents/*/BRIEF.md` and `model/CLAIMS.json` first.
- Start each agent's work from the best prior artefact (qualify, extend, stack), not from scratch.

**Step 2. While wave 1 runs (lead).**
- Wait on completion notices (§5.3).
- Build the next harness, e.g. a stage-isolating start state or a reproduction set.
- Answer progress questions with `editstack status`.
- Do not re-run agents' evals. Do not launch your own variants on an agent's line (the lead's r1a_g105 runs on latent's
  r1a, 11-07 03:23).

**Step 3. An agent finishes.**
- Read its REPORT.md first line and the board. Then decide:
  - its stage is solved → stop, or re-target it with a follow-up (§5.2);
  - it found the next bottleneck → re-target it to that bottleneck, as camfix was at 11-09 18:13 ("REACH is now 92–100%
    … make the GRASP the next target");
  - it ended null → write the negative result to MODEL_MAP, and free the slot.
- Fill the freed slot at once. Don't hold the queue for a go-ahead (§5.6).

**Step 4. Two stage owners both qualify. Merge (lead).**
- Rebase one chain on the other (PARENT), re-qualify on ≥ 3 seeds plus held-out, and check for double correction.
  - Shadow's radial fix on top of p17gshr pulled 1.2–1.8 cm short.
  - Neuron collisions: approach caught L15 779/1893 itself.
- Do leave-one-out on chain links before promotion. Roll was kept although neutral. In the hybrid project, 2 of 7 edits
  carried all of the main-suite gain.

**Step 5. Before promotion or publishing numbers. Independent reviewer (frontier).**
- Give it the board, the run dirs and the claim list. Promote only after it passes.

**Step 6. A new failure appears from outside (real robot, user video).**
- Lead first: reproduce it in sim as a fixed set. Then one stage owner owns that set (camfix, seek).
- The real arm is the lead's, after promotion, with a written expected result.

**Step 7. Stop.** The fleet is done when:
- the target is met on held-out, or
- every stage owner has ended null twice on its stage with no new causal sentence.

Then run analysts (frontier, 3–5 in parallel, one source each) and a reviewer for the retrospective.

**Never:**
- run a side project that loads models in the same RAM pool (the hybrid project cut throughput about 4x);
- run edit-maintenance agents (hybrid "maint" agents: 9 runs, mostly re-fitting what training erased);
- run a diagnose-only swarm that may not edit or score.

---------------------------------------------------------------------------------------------------------------

## 5. Spawn mechanics and lifecycle

### 5.1 Which spawn tool

| mechanism | seen here | use it for | notes |
|---|---|---|---|
| Native `Agent` tool (Claude Code) | 303f0667: 8 spawns, background by default, completion arrives as a task-notification with tokens/tools/duration | analysts, reviewer, short doc jobs inside one session | Background is the default. Pass `model` (`opus` / `sonnet` / `haiku`) explicitly. Continue an agent with `SendMessage`, not a new spawn. Start parallel agents in ONE message. |
| T3 `delegate_task` | lead 19, hybrid 35, Codex 22 | long-running stage owners that the user watches as their own threads | Set `target: {providerInstanceId, model, options: {effort}}`. Follow up with `t3_thread_send` (`mode: auto` / `steer` / `queue`). Completion arrives as a "Delegated task … [completed]" message. Stop with `task_cancel` / `t3_thread_interrupt`. |
| Codex parent with Claude children | 1 thread, 27 Haiku children | not recommended for this task | delegation failed late ("parent_not_active; no child was started"). The parent's context and the children's were never shared. |

- **Foreground vs background.**
  - Foreground only when your next step needs the result and there is nothing else to do. Otherwise background.
  - Stage owners, analysts and doc agents ran in the background throughout.
  - Two Haiku children returned "the run is still going, I'll wait" while their background job ran (28f8aee7,
    a3ed2742), so the parent had to continue them. Rule for children: **your final message is the result, never a
    status. Wait for your own jobs before ending.**

### 5.2 Follow-up vs new delegation
- **Follow up (`t3_thread_send` / `SendMessage`)** when the agent's context is still useful: same stage, new priority,
  new rule, new data, resume. The lead's 26 follow-ups were:
  - a metric change broadcast to 4 agents (11-07 02:52);
  - RAM rules to 5 agents (03:22);
  - the MODEL_MAP requirement (06:28);
  - resumes after network drops;
  - re-targeting camfix to grasp (11-09 18:13);
  - real-arm data to seek (11-10 00:10).
- **New delegation** when the stage changes or the context is mostly stale. Approach and grasp were new agents because
  the split into REACH and grasp changed the metric. They inherited wave 1's results by path, not by context.
- Send each follow-up once. Camfix got the same user goal twice at 16:53, once as `steer` and once as `auto`.
- Use `steer` for "change course now", `auto` for information, and `queue` for "after the current step" (hybrid: the
  ENDEDIT requirement was queued, 11-08 05:33).

### 5.3 Waiting without polling
- Wait on completion events: task-notifications, "[completed]" messages, `t3_background_wait`, `editstack wait`.
  Between events, do lead work (next harness, merges), or end your turn.
- Measured cost of polling:
  - the lead made 77 agent-folder reads, 24 `slots.py`, 33 `bench.py` and 27 `sleep` calls;
  - research agents ran up to 95 sleep commands each;
  - the hybrid orchestrator ran 97 background waits plus about 44 sleep loops and was active 5% of its 45.7 h.
- Every check pulls output into a context that is re-read on every later turn. Prefer one `editstack status` over
  many `tail`/`cat` calls.
- Agents waiting on their own sims: launch detached (`editstack eval … --detach`), then `editstack wait`. Never
  `sleep N; tail` loops.

### 5.4 The 2-hour background limit and detached runs
- Background shell jobs are killed after 2 h. That includes jobs that are only queued for RAM, and wait loops: the
  hybrid lost suite jobs at 11-08 03:40 and 04:18, and a waiter at 16:55.
- Start every eval that can exceed ~90 min (including queue time) fully detached (`nohup`/`setsid`, or
  `editstack eval --detach`). It writes a `DONE` marker; wait on the marker.
- Evals resume on re-run: finished episodes are kept.

### 5.5 Network drops: resume, don't relaunch
- Drops killed turns in approach (11-07 07:50), camfix (twice, 11-09 19:54), seek (11-09 22:55) and the hybrid
  (user interrupt, 11-08 05:30). Detached sims kept running every time.
- The lead's resume message that worked (fill in the bracketed parts):

  > Your previous turn died from a network error, not from your work. Resume where you left off. Your sims kept running:
  > `<run dir>` has `<k/n>` episodes and `<process>` is still going. Don't relaunch what is still running; wait for it.
  > Don't restart work that is already done. Check `agents/<name>/` for what you already wrote. The original brief still
  > applies. Next: `<1–3 concrete steps>`.
- The agent then checks running processes and run dirs **before** launching anything.

### 5.6 Go-aheads and defaults
- At 03:46 (11-07) the lead wrote: "I'm still waiting for your go-ahead to stop servo and pancost and start the approach
  and grasp agents". It asked again at 03:52.
- At 05:02 the user asked what the agents were waiting for (and raised the RAM allowance). At 05:03 the lead
  acknowledged they were mostly waiting on the lead itself.
- Cost: **77 min** with three research agents finished and the RAM pool mostly idle (servo alone, ~18 GB).
- Rule for every role: when a choice comes up, state the default you will take and take it in the same turn. Ask only
  for things that are irreversible or outside the brief (hardware, deleting others' files, changing the success
  definition).

### 5.7 Cancel
- No agent was ever cancelled. The lead said "I'll stop servo and pancost" (03:43) and never did.
- Cancel (`task_cancel` / `TaskStop`) when the agent's stage is gone, when it duplicates another agent, or when it holds
  slots for a line that has been ruled out.
- Before cancelling, send "write REPORT.md now, then stop", so the negative result reaches MODEL_MAP.
- Never kill another agent's sim processes without telling it. The lead killed latent's ID runs twice (03:22); latent
  never finished its ID check, and approach had to redo it.

---------------------------------------------------------------------------------------------------------------

## 6. Behaviour contracts

### 6.1 Every agent
Must:
- read the brief, `AGENTS.md` and the relevant MODEL_MAP lines before acting;
- write `agents/<name>/REPORT.md` from the first result on, with the first line = the current result, and update it
  after every result;
- go through slots for every model process, with ≤ 2 per agent. Count footprint, not RSS;
- detach runs that may exceed 90 min;
- state a default and proceed;
- end with the fixed final message (§6.8).

Never:
- write outside your folder, except: your own `edits/<name>_*` variants, append-only MODEL_MAP.md lines,
  `editstack learn` (LEARNINGS + strategy evidence logs) and runs produced through the CLI;
- bypass slots;
- touch hardware;
- copy checkpoints;
- delete or prune anything you did not create (the hybrid orchestrator pruned checkpoints at 98% disk; analysts
  report it deleted another agent's captures);
- run any scorer you have not read (`eval_output.py` printed a fake "correctness score");
- end with a status instead of a result.

### 6.2 Lead
Must:
- do Phases 0–2 yourself;
- build the staged metric and fixed suites before delegating;
- name the model and effort in every delegation;
- keep one chain;
- promote only via `editstack promote` after review;
- relay user requirement changes to every running agent in one broadcast (metric change, 02:52; RAM, 03:22);
- fill free slots at once.

Never:
- poll;
- re-run an agent's evals, or launch your own variants on its line;
- run a side project in the same RAM pool;
- hold a ready agent for a go-ahead you don't need;
- kill an agent's processes without a message.

### 6.3 Stage owner
Must:
- own exactly the stage in the brief and report other stages only as numbers;
- start from the named prior artefacts;
- verify offline (exactness, κ before/after, first-chunk aim) before any closed-loop sim;
- reserve neurons/channels as soon as you pick them: `editstack claims reserve <variant> <tensor> <axis> <idx>`, and
  `ctx.claim(...)` in edit.py (approach and grasp both used L15 1893);
- evaluate with `editstack qualify <variant>` (seed 0 + paired early stop, then the qualification seeds); a variant
  counts only when the board says qualified;
- write negative results to MODEL_MAP;
- report paired standard errors, and call a result within noise if it is (grasp: roll "within noise").

Never:
- work on another agent's stage;
- start fast-tier children that load the model (your 2-process budget includes them);
- promote;
- re-target yourself to a new stage without telling the lead in REPORT.md.

### 6.4 Doc / render agent
Must:
- write only under the one folder named in the brief;
- render only existing runs or the named variant, at ≤ 2 processes through slots;
- check rendered outcomes against the saved scores and report any mismatch (a free determinism check).

Never:
- modify code, edits, exports or other agents' folders;
- start sims for new variants;
- redo another doc agent's pass. Read the target folder first; if it exists, extend it.

### 6.5 Reviewer / verifier
Must:
- check every claim against a primary source (transcript, run dir, weights);
- give each claim a verdict (OK / WRONG / UNSUPPORTED / CONFOUNDED) with its evidence;
- re-derive the numbers, not copy them.

Never:
- accept the brief's framing as evidence;
- edit the artefacts under review;
- run sims or load models unless the brief says so.

### 6.6 Read-only analyst
Must:
- parse logs with scripts, never dump whole files;
- give each number its source (session id + time, or path);
- separate measured from inferred.

Never:
- write outside the scratchpad;
- state a causal reading without naming its confounds.

### 6.7 Fast-tier explorer
Must:
- do exactly the one computation in the brief;
- pass the reproduction gate before reporting anything;
- state units and sign conventions explicitly;
- finish with a result, not a status.

Never:
- interpret beyond the computation;
- propose edits;
- load the model unless the brief allots a slot;
- start background jobs that outlive its turn.

### 6.8 Final message format (every role)
```
RESULT: <one line, with numbers and seeds, or "null: <why>">
ARTEFACTS: <REPORT.md path>; <variant names / run dirs / files written>
CLAIMS: <neurons/tensors used, or none>
OPEN: <what remains, the default you would take next>
MODEL: <model id and effort you ran on>
```

---------------------------------------------------------------------------------------------------------------

## 7. Brief templates

The brief is the agent's only context. `editstack agent new <name> --stage <s> --target "<numbers>" --parent <v>`
writes the skeleton to `agents/<name>/BRIEF.md`. Fill in the rest, then delegate with "Read agents/<name>/BRIEF.md and
AGENTS.md, then do it."

**Length:** the briefs that worked were 2.7–4.9k characters. Length did not separate success from failure; content
did.

**Check before sending:**
- digits and spaces survive (the Haiku briefs reached the children as "still5/10 with88332");
- every path is absolute;
- the model and effort are set in the delegation call.

### 7.1 Stage owner
```
You are "<name>", the owner of one measurable quantity on a weight-editing task. Folder: <abs path>/agents/<name>/
(write only here, your edits/<name>_* variants, append-only model/MODEL_MAP.md and `editstack learn`). Read AGENTS.md and agents/<name>/BRIEF.md; read MODEL_MAP lines <a-b>.

STAGE YOU OWN: <stage>, measured by <staged label, e.g. REACH = P+R> on <suite>. Other stages belong to <agents>;
report them only as numbers.
WHY (causal sentence): <one sentence with a number and its evidence path>.
START FROM: <prior artefacts: variant, REPORT, construction>. Goal 1 is usually "qualify/extend <artefact>".
GOALS, in order: 1. ... 2. ... 3. ...
TARGET: <numbers over >= 3 seeds> on <suite>; no regression on <guard suite> (paired SE).
ALREADY TRIED (do not repeat): <MODEL_MAP negative results>.
HARNESS: <stage-isolating start state or reproduction set if any>. Verify offline before any sim.
BUDGET: <= 2 model processes via slots; detach runs > 90 min; claim neurons before eval.
DEFAULTS: if a choice comes up, take <default> and say so in REPORT.md. Don't wait for a go-ahead.
DELIVER: REPORT.md updated as you go (first line = result); variants on the board; final message in the fixed format.
```
**Filled example (approach, e5bc0951, 2,757 chars, produced p17gs → p17gshr).** Shortened; the lines that did the
work:
> You own ONLY POSITIONING: getting the gripper tip to the cube (staged benchmark REACH = outcomes P or R). Whether it
> then grips is another agent's job ("grasp"); ignore PICK except to report it.
> 1. **Qualify the reach fix.** Run the in-distribution check for agents/latent/ed/r1a … for seeds 0 and 1 …
> 2. **Remove pan×1.7's in-distribution over-aim** with a NONLINEAR pan gain … Pancost's measurements are in
>    agents/pancost/REPORT.md … A good starting point is circuit's exact, drift-free neuron-pair construction
>    (agents/circuit/circ.py) …
> Verify offline before spending sim time …
> Success criteria: Near-lateral REACH ≥ 34/36 over seeds 0–2. In-distribution REACH ≥ 95% AND ID PICK not below the
> export's level …

**Grasp (659c3fba)** added the most valuable line of all:
> **Isolate grasping from positioning.** Build your own harness … that starts each episode with the arm ALREADY
> hovering above the cube.

That gave a 42-start harness and the roll diagnosis (49/69).

**Camfix (a6b62e9c, 4,917 chars)** was the most productive per hour (1,173 episodes in 4 h, best-thus-far). It worked
because the brief contained:
- what changed, with measured numbers (FOV 63.97° → 42.6°, 80.7° → 59.3°);
- a fixed goal on fixed suites;
- numbered hard rules;
- a suggested plan starting with a baseline table;
- a report format capped at ~600 words.

### 7.2 Contrast: briefs that underperformed
- **latent / circuit / pancost (wave 1, 2.1–2.6k chars).**
  - The brief opened with "Theory to test: …" and asked for "≥ 2 seeds" on the near set and in-distribution, scored as
    PICK. That metric could not see a reach fix, so two real reach fixes read as nulls.
  - Latent and circuit had the same target (reach) at the same time.
  - The fix is not to drop the hypothesis. Give a metric that sees it, and one owner per quantity.
  - Servo's brief, also "your theory to test", worked: its κ measurement and its target (ep5/ep11 reach failures) were
    visible.
- **Haiku children (2.8–6.1k chars).**
  - Most of each brief was prohibitions ("no regression or fitting", "do not edit model weights", "do not run full
    evaluation suites") and filesystem fences around one prescribed diagnostic.
  - No brief gave a target the child could move. The only possible output was a verdict on someone else's hypothesis.
  - Numbers were whitespace-stripped.
  - Result: 0/18 accepted; 14/18 corrected.
- **Hybrid "maint" briefs.** "Check survival of last cycle's circuits; refit". These produced re-fits of what training
  kept erasing.

### 7.3 Doc / render
```
Documentation task in <abs project path>. Write ONLY under <folder>. Don't modify code, edits, exports or agents/.
Read <folder> first; extend what exists, don't redo it.
Render: <variant(s)>, <episodes/placements>, seed <s>, <= 2 processes via slots. Check each rendered outcome against
<saved run dir> and report mismatches.
Output: <page/GIF spec, size limits>. Final message: paths + a 5-line summary + MODEL.
```
- Example: 18b03fd0 (side-by-side base vs p17gshr). 6 clips, scored with `bench.py --detail`, 46 min wall, 5 min
  active, 28 tools.

### 7.4 Reviewer
```
Independent audit. Claims to check: <file(s) or list>. Primary sources: <transcripts, run dirs, board, weights>.
For each claim: verdict OK / WRONG / UNSUPPORTED / CONFOUNDED + the evidence (path, id+time). Re-derive numbers.
Read-only except <scratch dir>. No sims, no model loads. Output: a table + the 5 changes that matter most.
```

### 7.5 Analyst
```
Read-only retrospective of <one source>. Parse with scripts; never dump whole files. Report: <table spec>,
each number with its source; separate measured vs inferred; name confounds. Write only <scratch file>.
Final message: <= 300 words + the file path.
```
- Launch 3–5 at once, one per source (lead transcript, sub-agent reports, side project, early phase), then a reviewer.

### 7.6 Fast-tier explorer (only for checkable jobs)
```
One mechanical job. Inputs: <paths>. Compute: <exact procedure>. Units: <units>; sign convention: <convention>.
Gate: first reproduce <reference output> bit-exactly; if it fails, stop and report that.
Write only <folder>. Output: <table/JSON schema>. Do not interpret. End with the result, not a status.
```

---------------------------------------------------------------------------------------------------------------

## 8. Cost accounting

| role (n) | wall min (mean) | active min | tools (mean) | output tok | cache-read tok | outcome |
|---|---|---|---|---|---|---|
| lead (1) | 7,019 | 557 | 746 | 0.86 M | 317 M | pad0, pan17, tooling, merges, promotion |
| stage owners (9) | 2,265 (252) | 707 | 1,339 (149) | 1.28 M | 330 M | 5/9 in best lineage, 2 more as method/spec |
| doc, Opus (9) | 167 (19) | 84 | 356 (40) | 0.38 M | 37 M | 9/9 delivered, 1 duplicate |
| doc, Sonnet (2, hybrid) | 19 (10) | 10 | 39 (20) | 0.03 M | 2.6 M | 2/2 delivered |
| Haiku explorers (27) | 233 (9) | 233 | 941 (35) | 2.55 M | 111 M | 0/18 accepted, 14/18 corrected |
| analysts (4, this retro) | 22 (5.5) | — | 129 (32) | 663k total tokens | — | notes; several over-readings |
| reviewer (1) | 10.6 | — | 48 | 209k total tokens | — | caught 5 errors in the docs |
| hybrid orchestrator (1) | 2,746 | 148 | 532 | 0.38 M | 194 M | ran 12 cycles; in-training edits ≈ control |
| hybrid children (35) | 5,271 (151) | 1,219 | 3,025 (86) | 2.61 M | 514 M | 7/17 new edits kept; end-edit beat all |

Trade-offs:
- **Long-lived orchestrators cost the most tokens.** The two orchestrators read 511 M cache tokens. The main lead alone
  (317 M) matched all 9 main-project stage owners (330 M). Keep the lead's context small: don't poll, give renders to
  doc agents, and run no side projects. About 8 lead turns at 500k context re-read as much as one whole doc agent
  (mean 4.1 M).
- **Stage owners are cheap in tokens relative to their wall time** (~140k output over 4 h). The sim is the cost. A
  downgrade saves little, and one wasted 2-seed closed-loop check costs 1–2 h of a scarce slot.
- **The fast tier was not cheap here.** 27 Haiku sessions emitted 2.55 M output tokens, twice the output of all 9 Opus
  stage owners, plus the frontier parent's review time, and produced no accepted edit. The per-token price is lower;
  the cost per accepted result was unbounded.
- **Reviewers and analysts are the best value per token** (100–235k tokens, 3–11 min each). One reviewer pass
  corrected rules that would otherwise have steered every future agent wrong.
- **Accuracy vs cost rule [judgement]:**
  - Spend frontier tokens where an error costs sim hours or a wrong rule (lead, stage owner, reviewer, analyst).
  - Spend mid-tier tokens where errors are visible and cheap to fix (renders, pages).
  - Spend fast-tier tokens only where a checker rejects errors automatically.

---------------------------------------------------------------------------------------------------------------

The corrections found while writing this file have been applied to LESSONS.md (the single source of numbers).
