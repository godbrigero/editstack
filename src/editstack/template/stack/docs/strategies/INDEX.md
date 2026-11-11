# Strategy library: index and selection guide

Use this after ANALYSIS_GUIDE.md has given you a measured symptom. It supersedes the symptom table in
APPROACH_PLAYBOOK.md (that table is the short version; the files here have the full procedure, math, numbers and
evidence). Recipes for the primitive ops are in EDIT_COOKBOOK.md; the evidence narrative is in LESSONS.md; how to
edit without a sim is in NUMBERS_ONLY_EDITING.md; how to split work across agents is in AGENT_FLEET.md.
Update a file after every run that touches it: HOW_TO_UPDATE.md.

Status: **proven** = repeated closed-loop gain beyond noise or an exactness proof plus a gain; **partial** = mechanism
verified but the score gain is within noise or base-dependent; **failed** = tried and worse/null (anti-strategy);
**untested idea** = a concrete procedure with no run yet.

Path prefixes in evidence logs: `OLD` = ~/Documents/smolvla-weight-edit-6000 (SmolVLA SO-101, Nov 5–10
2026), `HYB` = ~/Documents/hybrid-train-smolvla, `CODEX` = ~/Documents/yellow-cube-manual-weights-20261104.
"lead b078391b" = the lead transcript in ~/.claude/projects/-Users-godbrigero-Documents-astra-direct-weight-edit/.
"scratchpad …" = secondary retrospective notes (verified against primary sources where cited).

## All strategies

| id | name | status | pick it when | typical gain (past) | cost |
|---|---|---|---|---|---|
| [S01](S01-disconnect-unsupervised-inputs.md) | Disconnect never-supervised inputs (pad-zero) | proven | jitter inside chunks; padded/unsupervised channels fed at inference | 8/20 → 17/20, jitter 0.91 → 0.19; hybrid ENDEDIT 44/80 vs 37 | minutes, 1 tensor |
| [S02](S02-exact-output-translation.md) | Exact output-space translation (flow τ term) | partial | constant offset (closes N mm short; real-arm offset) | exact shift (τ-fit 1.7e-8); score gain within noise (17 → 19/20, 1 seed) | minutes, 4 tensors |
| [S03](S03-per-axis-affine-scale.md) | Per-axis affine scale, perception compensated | proven | error grows beyond the trained range on one axis; κ ≪ 1 there | near REACH 5/12 → 31/36; ID PICK cost 73 → 63/80 | minutes + exactness probe |
| [S04](S04-readout-ridge-servo.md) | Rank-1 ridge servo gain at the read-out | proven | feedback correction exists but too weak (κ ≈ 0.13) | targeted eps 1/6 → 5/6; hybrid 44 → 60/80 | 1 capture, seconds to fit |
| [S05](S05-gated-product-circuits.md) | Gated circuits in spare SwiGLU units | proven | need a nonlinear / phase- or band-gated term, zero in-distribution | p17gs near 35/36, ID 38/40; p17gsh ID 40/40 | capture + readouts, 2–6 units/term |
| [S06](S06-gate-a-fix-out-of-band.md) | Gate an existing fix to where it is needed | proven | a global fix costs in-band (over-aim 1.5×) | ID aim 1.53 → 1.03; ID REACH 37 → 38–40/40 | as S05 |
| [S07](S07-fold-paired-map-upstream.md) | Fold a paired-render map upstream (input shift) | proven | everything broke after a camera/sensor/preprocessing change | ID REACH 6/40 → 57–59/60; downstream edits survive | ~4 h incl. captures |
| [S08](S08-jacobian-geometry-to-joints.md) | Jacobian geometry → joint deltas | proven | you have a task-space error and need an output vector/direction | enables S02/S04/S05 | minutes |
| [S09](S09-controlled-scenes-and-chunk-probes.md) | Controlled scenes + single-chunk probes | proven | always; every measurement and offline verification | seconds per question vs ~50 s/episode | build once |
| [S10](S10-stage-isolating-harness.md) | Stage-isolating start states / reproductions | proven | stage failures confounded; a real failure doesn't reproduce | roll diagnosis 40/42 vs 33/42 in one run | 1–2 h to build |
| [S11](S11-rescore-stored-trajectories.md) | Rescore stored trajectories; staged metric | proven | metric changes; an edit looks null | r1a "9/12 picks" → 23/24 REACH | seconds |
| [S12](S12-ablate-chain-links.md) | Ablate chain links before promotion | partial | chain > 2 links; links added on small marginals | hybrid: last 4 circuits added 0 (60 → 60/80) | n extra paired runs |
| [S13](S13-gated-servo-for-unaimed-axis.md) | Gated servo for an unaimed axis (roll) | partial | failures dominated by one secondary axis with κ ≈ 0 | misalignment 20.3° → 15.9°; picks within noise | 2 units |
| [S14](S14-gated-correction-of-sim-artifact.md) | Gated correction of a sim-only artifact (shadow) | partial | performance drops without the artifact, with a systematic bias | sp6 OFF ID REACH 31 → 40/40; hurts p17gshr | high (~9 h) |
| [S15](S15-gated-seek-circuit.md) | Gated seek ("search until seen") | partial | policy stalls when the target leaves a view | repro REACH 10/24 → 22/24; neutral elsewhere (2 seeds) | 2 units, ~3 h |
| [S16](S16-perception-level-rank1-rewrite.md) | Perception-level rank-1 o_proj rewrite | partial | the model reads the wrong function of a decodable quantity | reach error −3.1/−4.1 → +0.5/0.0 cm; picks null; in-band cost | 1 capture |
| [S17](S17-replan-horizon-tweaks.md) | Re-plan / horizon tweaks | failed | — | near PICK 21/24 → 15/24 (n25), 9/24 (n10) | ~1 h queue |
| [S18](S18-editing-during-training.md) | Editing during training | failed | — | 39/80 = control 37/80; end-edit 60/80 | ~34 h of interventions |
| [S19](S19-global-output-gain.md) | Global output gain | failed | — | ×1.2: 6/20 vs 8/20 | — |
| [S20](S20-hardcoded-controller.md) | Hard-coded controller (2×10⁸ values) | failed | — | 2/5 → 2/10 on counter-cases; rejected | ~1.4 h + restart |
| [S21](S21-interpretability-hunt-first.md) | Head/neuron interpretability hunt first | failed | — (use it later, to locate a site) | 14 h plateau at 5/10 (confounded comparison) | hours |
| [S22](S22-diagnose-only-swarms.md) | Diagnose-only agent swarms | failed | — (see AGENT_FLEET.md) | 0/18 tracks accepted | 4 h of children |
| [S23](S23-internal-gain-and-input-warps.md) | Internal layer/sub-space gains, input warps | failed | — | radial κ down or unchanged; drift 2–8 cm | — |
| [S24](S24-naive-bias-nudges.md) | Naive bias nudges / time-feature shift | failed | — | 6.2/8.5/−14.9 vs 1.5/1.5/−3.1 target | — |
| [S25](S25-temporal-ensembling.md) | Temporal ensembling (execution) | untested idea | boundary jumps dominate after S01 | unknown | k× inference |
| [S26](S26-task-vector-lowrank-merge.md) | Low-rank task-vector merge | untested idea | a related checkpoint has the missing skill | unknown | SVD per tensor |
| [S27](S27-norm-scale-recalibration.md) | Norm-scale recalibration (diagonal S07) | untested idea | input shift ≈ per-channel rescale | unknown (≤ S07) | captures |
| [S28](S28-mean-shift-bias-correction.md) | Mean-shift bias correction (offset-only S07) | untested idea | first cheap try after an input shift | unknown | seconds |
| [S29](S29-prune-dead-units.md) | Prune dead units, verified | untested idea | before claiming "spare" units for S05 | neutral by design; safer unit pool | offline |
| [S30](S30-quantization-aware-edits.md) | Quantization-aware edits | untested idea | small deltas in bf16/fp16 tensors | unknown | offline |

## Decision flow

Always: S09 (probes) and S11 (staged metric, stored trajectories) are in place before anything else; harness audited
(HARNESS_AUDIT.md). Prefer the highest rung: exact algebra (S01–S03, S07) > closed-form fit (S04, S16) > gated circuit
(S05, S06, S13–S15). Before stacking any link ask: **does an earlier link already correct this quantity?**

| symptom / situation (measured) | candidates, in order | first confirm with |
|---|---|---|
| jitter inside chunks | S01 → (boundary jumps remain) S25 | loss mask + zero-columns chunk probe (d² ↓) |
| constant signed offset in task space | S08 → S02 | flat signed error vs task variable |
| error grows with distance from the trained range on one axis | S03 → S06 (if ID cost) | κ per axis and region; R² of the quantity high |
| correction exists but too weak (κ ≈ 0.1–0.3), right direction | S04 → S06 (gate it) | closed-loop state/image κ split; read-out separability |
| a fix helps out-of-band, costs in-band | S06 (built with S05) | per-region chunk-1 gain before/after |
| one secondary axis never aimed (κ ≈ 0), taxonomy points at it | S10 (upper bound) → S13 | harness aligned vs natural |
| reaches but wrong function of a perceived quantity (|y| vs r) | S04/S05 at the read-out first; S16 only if read-out can't see it | decodability of the shortfall; patching |
| everything broke after camera/sensor/preprocessing change | S28 (cheap) → S27 → S07 | paired teacher-forced captures; fraction of shift left |
| depends on a sim-only artifact | measure on the CURRENT chain; S14 only if a systematic bias remains | paired renders ON/OFF, closed loop |
| stalls / no attempt when the target leaves a view | S10 reproduction → S15 | per-camera visibility (pixels) vs chunk progress |
| a related checkpoint has what you need | S26 (first in chain) | offline chunk comparison of both |
| want spare units for a circuit | S29 → S05 | importance on all phases |
| exported model weaker than the offline probe | S30 | probe on cast weights |
| chain has many links / ready to promote | S12 → promote | forward stages + leave-one-out |
| tempted to: re-plan more often, scale all outputs, nudge a bias, scale a layer, hand-code a controller, edit while training, hunt heads first, spawn read-only swarms | read S17–S24 first | — |

Order when several fit: exact single-tensor fixes first (S01, S02), then axis scaling (S03), then read-out fits (S04),
then gates (S05/S06), then situational circuits (S13–S15); input-shift repairs (S07/S27/S28) go UPSTREAM of everything
and do not require refitting downstream edits. Re-qualify after every link (AGENTS.md Phase 4).

## Corrections to other docs

Found by checking primary evidence (and an independent audit, scratchpad review/REVIEW.md §1). The other docs were not
edited by this writer.

1. **cf2 "ID reach 6/40 → 59/60"** (LESSONS.md table, APPROACH_PLAYBOOK.md row "everything broke…", the docstring of
   edits/_example/edit.py): cf2 is **58/60** ID REACH (near 35/36); 59/60 is cf3p25. cf1 (500 scenes) already had 57/60.
2. **"~2 days of tuning on wrong views"** (LESSONS.md "Harness first"): the circuits carried over unchanged — cf2 =
   p17gshr + upstream maps. Real cost: finding the mismatch late plus ~4 h of camfix work (S07).
3. **"re-plan every 10 or 25 steps, temporal ensembling (worse: 9/24 and 15/24 vs 21/24)"** (APPROACH_PLAYBOOK.md):
   9/24 and 15/24 are the horizon test (n_action_steps 10/25 on cf3p25, measured cameras, near PICK, 2 seeds).
   Temporal ensembling was never measured → S25 untested.
4. **Roll servo "+3 near picks (p17gshr)"** (APPROACH_PLAYBOOK.md): within noise (net over 4 bases near +3, ID −2;
   lead: "alignment improves, picks don't") → S13 partial.
5. **deep8 "19/20" as a win** (LESSONS.md, APPROACH_PLAYBOOK.md, ANALYSIS_GUIDE.md "74 min"): one seed, within noise.
   The exact method is the lesson; pad-zero's 8 → 17/20 with jitter 0.91 → 0.19 is the real win.
6. **"Interference … shared a neuron (caught via a claim list)"** (LESSONS.md): there was no claim list; approach's
   REPORT caught 779/1893, and the MODEL_MAP registry wrongly listed 1893/1701 as free.
7. **Hybrid "editing the finished checkpoint once: 60/80"** (LESSONS.md, APPROACH_PLAYBOOK.md): true, but pad-zero +
   one read-out servo (m2) already gave 60/80; the four later circuits added nothing on the main suite; ENDEDIT costs
   +27 % held-out loss.
8. **"Stage owners beat hypothesis testers"** (LESSONS.md): confounded — sp6 (first best-thus-far) came from a
   mechanism brief, and wave-1 "nulls" were judged on PICK before REACH existed (S11). Supported version: give each agent
   a metric that can see its effect, a numeric target and a fixed suite; never two agents on one hypothesis.
9. **"14 h head/neuron hunt 5/10 vs 19/20 in 74 min"** (ANALYSIS_GUIDE.md, LESSONS.md): different models (step 2000 vs
   6000) and tests (strict 18 s contact vs held ≥ 1 s); not a controlled comparison (S21).
10. **"Diagnose-only swarms: 0 accepted"** (LESSONS.md, ANALYSIS_GUIDE.md): numbers right, cause confounded (Haiku,
    forbidden to edit/score, stuck parent) (S22).
11. **EDIT_COOKBOOK.md "Fold a linear map … fit.paired_map"**: the map that worked shrinks to the IDENTITY without an
    intercept, λ chosen per layer by CV, teacher-forced pairs; `fit.paired_map` shrinks to zero with an intercept (S07).
12. **"pan x1.7: near 5/12 → 31/36"** (LESSONS.md, APPROACH_PLAYBOOK.md) is REACH; under PICK it is 1/12 → 29/36, with
    an ID PICK cost of 73 → 63/80 that the docs do not state.
