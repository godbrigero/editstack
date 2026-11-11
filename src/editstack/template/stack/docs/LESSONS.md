# Lessons: the evidence behind AGENTS.md

Source: retrospective of the SmolVLA SO-101 yellow-cube project (Nov 2026), ~50 agent sessions, ~4,300 sim
episodes. Times UTC. Numbers are from run logs and reports; see each line.

## What produced the wins

| step | edit | how it was derived | result |
|---|---|---|---|
| jitter fix | `pad0`: zero 26 never-supervised padded action-input columns | read loss code (`losses[:, :, :6]`) + offline chunk probe | 8/20 -> 17/20, jitter 0.91 -> 0.19 |
| deeper grasp | `deep8`: exact translation of the action space by an 8 mm Jacobian step | MuJoCo Jacobian + flow-matching algebra (bias + time-feature terms) | 19/20 (74 min after start) |
| lateral reach | `pan x1.7`: exact affine rescale of one joint axis (command gain, perception compensated) | derived algebraically, verified offline (max diff 0.05) before any sim | near-lateral 5/12 -> 31/36 |
| image servo | `sp6`: rank-1 ridge fit amplifying the image term of the correction | closed-form fit on recorded activations | near 33/36 reach |
| gated gains | `p17gs/gsh/gshr`: pan gain, servo, hold, roll circuits in spare layer-15 units, zero in-band | exact product pairs gated on "planned - state" / out-of-band features | near 35/36 reach 32/36 pick; ID 40/40, 34/40 |
| camera change | `cf2`: per-layer linear maps folded into 16 attention outputs | paired renders (measured vs training cameras), ridge | ID reach 6/40 -> 59/60 |

Common factor: every winning edit was a deterministic, derived function of the weights (algebra, a closed-form fit,
or a gated circuit). None came from searching scales by trial and error.

## What the model map showed (the facts that made targeted edits possible)
- Cube position linearly decodable everywhere (R^2 0.87-0.99): perception fine, read-out at fault.
- Per-chunk correction gains far below 1 (pan 0.34-0.6, radial ~0.13, roll ~0.03).
- First move aimed by the front camera alone, reading forward distance not radius -> side cubes closed short.
- Images reach actions only through the expert's odd (cross-attention) layers.
- 49 of 69 reached-but-not-picked failures: wrist roll misaligned > 25 deg.

## Process lessons (with cost)
- **Harness first.** Sim cameras never checked against real until day 4: ~2 days of tuning on wrong views; the
  best variant collapsed to 6/40 under measured cameras. Also a stale copy of the sim code went unnoticed.
- **Structural defects before interpretability.** A parallel 14 h head/neuron/RoPE hunt plateaued at 5/10; the
  structural route got 19/20 in 74 min.
- **Stage owners beat hypothesis testers.** Agents owning REACH, GRIP, CAMERAS produced every best variant; the
  first wave's "test theory X" agents (latent, circuit, pancost) all ended "keep the baseline" (but circuit's
  exact neuron-pair method became the template for all later circuits).
- **Diagnose-only swarms fail.** 27 small read-only agents (avg 9 min, 35 tool calls): 0 accepted candidates;
  each re-derived the same context; probes too small to resolve signals; sign errors.
- **Seeds.** Two seeds misled twice (23/24 -> 31/36; 24/24 -> 31/36). Qualification now needs >= 3 ood seeds.
- **Winner's curse.** Dozens of variants chosen on the same 32 placements; leaders separated by 1 episode. Hence
  the held-out suite.
- **RAM.** Real footprint ~8 GB/process (ps showed 1-4 GB); 15 processes hit ~110 GB. Slots fixed it. Throughput
  peak 250-330 episodes/h at 6 processes; useful concurrency ~3 agents x 2 processes.
- **Waiting.** Agents idle 60-85% of wall time on sims; lead polled (~75 result checks, 27 sleeps) and grew to
  ~780k tokens context. A training side-project in the same RAM pool cut throughput 4x.
- **Horizon/re-plan tweaks** (replan every 10/25 steps) made picks worse (9/24, 15/24 vs 21/24); fully reverted.
- **Edit during training is self-defeating.** Hybrid project: editing during training 39/80 reach = control
  37/80; editing the finished checkpoint once: 60/80. Training cancels active edits; frozen circuits' inputs drift.
- **Interference** only where two edits corrected the same quantity, or shared a neuron (caught via a claim list).
- **Network drops** kill agent turns; detached runs + "resume, don't relaunch" saved the work.
- **Fake metrics.** A script named eval_output.py printed a "correctness score" that only counted its own
  invocations. Never trust a scorer you have not read.
