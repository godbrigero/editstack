# S10 — Stage-isolating start states and failure reproductions

Status: proven
Last updated: 2026-11-10

## When to use
- Failures of a later stage (grasp) are mixed with failures of earlier stages (reach), so a fix to one stage is
  invisible or confounded in the full-episode score.
- A real-world failure does not reproduce on the standard sets (seek: every standard cube was in the wrist view
  after chunk 1).
- You need many cheap episodes of ONE stage to test a hypothesis (closing depth vs jaw alignment).

## Preconditions & confirming probes
- The sim lets you set the initial robot configuration (`arm.initial_qpos` before `driver.reset()`) and inject
  perturbations at execution time without lying to the policy (state stays honest).
- A staged outcome metric (bench.py: P/R/s/l/w/h/N) so the stage you isolate is scored on its own.

## Procedure
1. **Start-state harness** (`OLD/agents/grasp/hover.py`): copy of the eval; each episode starts at the demo IK
   hover (tip 10 cm above the grasp target, gripper 95 % open), 10 s episodes.
2. **Controlled factor**: hand yaw = cube yaw + dyaw. Sets of 6 positions × 7 yaws = 42: `al10` (aligned jaws) vs
   `nat10` (hand yaw 0 = what the model naturally does).
3. **Fidelity check vs real rollouts** before trusting it: R rate 9/42 = 21 % vs 20 % closed loop; same R signature
   (gripper 3–5.5 after the close, cube pushed 2.5–3.4 cm); same close timing.
4. **Reproduction knobs** (`OLD/agents/seek/evalseek.py`): `--c1-scale g` (first chunk executes only g× its pan move;
   state stays honest), `--pan-off K` (encoder offset); log what each camera sees (corner fraction AND yellow-pixel
   count — an open jaw can hide the cube with all corners in frame).
5. Use the harness to choose between hypotheses, then validate the chosen edit on the full sets.

## Verification
- The harness reproduces the closed-loop failure rate and signature (above). Re-check after changing the base model.

## Expected effect & cost
- Grasp harness: aligned 40/42 vs natural 33/42 → jaw roll is the cause; deeper close 33/42 (rejected in one run).
  Upper bound for a roll fix established in one run.
- Seek reproduction: near with chunk-1 pan × 0 → REACH 10/24 (cf3p25), the target for the seek circuit (S15).
- Cost: ~1–2 h to build; each 42-episode run ≈ the cost of a near set.

## Failure modes / interactions with other strategies
- A harness that is not checked against real rollouts measures a different problem.
- A reproduction knob can create failures the real system doesn't have; tie it to a real observation (seek used the
  real arm's pan after chunk 1).

## Evidence log
- 2026-11-07, SmolVLA sp6, grasp hover harness: al10 40/42, nat10 33/42, deep4 33/42. Evidence: OLD/agents/grasp/REPORT.md §1; OLD/agents/grasp/{hover.py,sets.py}.
- 2026-11-09, cf3p25, seek reproduction sets: near c1-scale 0 10/24 REACH; rf c1-scale 0.35 8/16. Evidence: OLD/agents/seek/REPORT.md §1.
