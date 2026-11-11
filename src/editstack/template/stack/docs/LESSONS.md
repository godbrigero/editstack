# Lessons: the evidence behind the guides (single source of numbers)

Source: the SmolVLA SO-101 yellow-cube project (Nov 2026): ~50 agent sessions, ~4,300 sim episodes, an
independent evidence audit, and three evidence-based guide writers. Other guides cite this file; when a number
changes, change it HERE. Corrections from the audit are marked **[corrected]**.

## What produced the wins

| step | edit | how it was derived | result | strength of evidence |
|---|---|---|---|---|
| jitter fix | `pad0`: zero 26 never-supervised padded action-input columns | read the loss code (`losses[:, :, :6]`) + an offline chunk probe | 8/20 -> 17/20 held >= 1 s; jitter 0.91 -> 0.19 | strong (large effect) |
| deeper grasp | `deep8`: exact translation of the action space by an 8 mm Jacobian step | MuJoCo Jacobian + flow-matching algebra (bias + time-feature terms) | 17/20 -> 19/20 | **weak: one seed, within noise [corrected]** |
| lateral reach | `pan x1.7`: exact affine rescale of one joint axis (command gain, perception compensated) | algebra, verified offline (max diff 0.05 raw units) before sim | near-lateral REACH 5/12 -> 31/36; **cost: ID PICK 73 -> 63/80** (in-band over-aim 1.5x, later removed by gating) | strong |
| image servo | `sp6`: rank-1 ridge fit amplifying the image term of the correction | closed-form fit on recorded activations (servo agent, a *mechanism* brief) | near 33/36 REACH; first best-thus-far | moderate |
| gated gains | `p17gs/gsh/gshr`: pan gain, servo, hold, roll circuits in spare layer-15 units, silent in-band | exact product pairs gated on cube-x / gripper state | near 35/36 REACH, 32/36 PICK; ID 40/40, 34/40 (training cameras) | strong for reach; **roll part: alignment improved, picks within noise [corrected]** |
| camera change | `cf2`: per-layer linear maps folded into the 16 expert attention outputs | paired renders (measured vs training cameras), ridge | ID REACH 6/40 (p17gshr under measured cameras) -> **58/60** (cf3p25: 59/60) **[corrected]** | strong; maps partly fit on eval placements (needs held-out) |

Common factor: every winning edit was a deterministic, derived function of the weights (algebra, a closed-form fit
or a gated circuit). None came from searching scales by trial and error.

## What the model map showed (the facts that made targeted edits possible)
- Cube position is linearly decodable everywhere (R^2 0.87-0.99), so perception was fine and the read-out was at fault.
- Per-chunk correction gains were far below 1 (pan 0.34-0.6, radial ~0.13, roll ~0.03).
- The first move is aimed by the front camera alone, reading forward distance rather than radius, so side cubes closed short.
- Images reach the actions only through the expert's odd (cross-attention) layers.
- The layer-15 MLP input linearly encodes every gate the circuits needed.
- 49 of 69 reached-but-not-picked failures had wrist roll misaligned by more than 25 deg.

## Process lessons
- **Check the harness early. [corrected]** The sim cameras were never checked against the real ones until day 4, and
  p17gshr collapsed to 6/40 ID REACH under measured cameras. But the circuits were NOT wasted: cf2 = p17gshr + maps
  folded upstream recovered 58/60. The real cost was the late discovery plus ~4 h of camfix work. A stale copy of the
  sim code also went unnoticed. Lessons: ask the user for one real frame per camera at the start, and fix input
  shifts by folding a map UPSTREAM of existing edits.
- **Structure before interpretability. [confounded, kept as judgement]** The structural route reached 19/20 in 74 min;
  a parallel head/neuron/RoPE hunt sat at 5/10 for ~14 h. The two used different checkpoints (step 6000 vs 2000) and
  tests (lenient 20-episode vs strict 10-case), and the slow line had also found pad-zero (0/5 on its test). pad0
  itself is a real win.
- **What agents need. [corrected]** "Stage owners beat hypothesis testers" was wrong as stated. sp6, the first best,
  came from a mechanism brief (servo). The first wave looked null only because it was scored on PICK: latent and
  circuit did fix reach (radial -3 cm -> ~0; latent's r1a scored 23/24 REACH once REACH was measured). The rule:
  every agent needs a staged metric that can SEE its effect, a numeric target and a fixed suite. Two agents on the
  same hypothesis is confirmed waste (latent + circuit). Details: AGENT_FLEET.md.
- **Diagnose-only swarms. [confounded]** The 27 Haiku children got 0 accepted candidates out of 18 reviewed, but
  their briefs forbade editing and scoring, and their parent was stuck at 5/10, so zero was guaranteed by design.
  This does not show that the fast tier is useless. It did well at bit-exact reproduction and arithmetic audits, and
  badly at interpretation (a reversed sign, a wrong axis, wrong units).
- **Seeds. [corrected]** Two seeds misled three times: r1a_xg 23/24 -> 31/36 (REACH), sph6 23/24 -> 32/36,
  grh20 24/24 -> 31/36 (PICK). Qualification therefore needs >= 3 seeds on the target suite.
- **Winner's curse.** Dozens of variants were chosen on the same 32 placements; leaders were separated by one episode.
  Hence the held-out suite.
- **RAM.** The real footprint was ~8-9 GB per process (ps showed 1-4 GB), and 15+ processes hit ~110 GB. **[corrected]**
  Throughput peaked at 326 episodes/h with ~4.7 busy processes and fell to 254 at ~6 (83 s per episode vs 33 s
  uncontended). Use `N_research = floor(min(P_ram, P_sat) / 2)`, which gives 2 here (3 at most).
- **Waiting. [corrected]** Agents sat idle 25-86% of wall time on sims (7 of 9 above 55%). The lead's ~780k-token
  context peak came during a hybrid-training side project (polling contributed). A side project in the same RAM pool
  cut throughput ~4x. The "agents waited ~80 min for a go-ahead" episode was the LEAD holding two unlaunched agents
  for 77 min.
- **Horizon / re-plan tweaks [corrected]** (re-plan every 10/25 steps) made picks worse: 9/24 and 15/24 vs 21/24.
  Fully reverted. Temporal ensembling was never measured.
- **Editing during training is self-defeating.** In the hybrid project, editing during training scored 39/80 REACH,
  the same as the control (37/80); editing the finished checkpoint once scored 60/80. Training cancels active
  edits, and frozen circuits' inputs drift. Corollary: pad-zero + one servo already gave 60/80, and the four later
  circuits added nothing on the main suite, so ablate chain links before promoting.
- **Interference** appeared only where two edits corrected the same quantity (shadow's radial fix on top of p17gshr)
  or shared a neuron. **[corrected]** No claim list existed: approach caught the 779/1893 collision itself, and the
  shared MODEL_MAP registry was itself wrong. Hence the automatic CLAIMS.json with derived conflicts.
- **Network drops** killed agent turns; detached runs plus "resume, don't relaunch" saved the work.
- **Fake metrics.** `eval_output.py` printed a "correctness score" that only counted its own invocations. Two agents
  shared its counter, and one read the jump to "2" as progress. Never trust a scorer you have not read.
- **Re-scoring.** When the success definition changed, the old project re-scored stored trajectories with no sim
  cost. Keep trajectories, and version the metric separately from the sim (`metric_version`, `editstack rescore`).
