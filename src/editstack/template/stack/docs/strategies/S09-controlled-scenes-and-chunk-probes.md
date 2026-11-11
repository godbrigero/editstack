# S09 — Controlled-scene rendering and single-chunk offline probes

Status: proven
Last updated: 2026-11-10

## When to use
- Always, before any sim rollout, to answer one question with one number in seconds: "does the output move the
  right way and by how much when input X changes?"
- To measure gains κ, hold behaviour, drift, side effects, and to verify every edit offline (rung a).

## Preconditions & confirming probes
- A renderer that produces a full observation for an arbitrary (arm configuration, object pose) without running a
  rollout, and a loader that runs one chunk with FIXED noise.
- Fixed noises: same noise for each pair being compared (paired differences cancel sampler noise).

## Procedure
1. **Scene renderer** (`OLD/scene.py`): `Scene.hover(x, y, yaw, h)` = IK pose with the tip h above the object,
   gripper open; `Scene.render(q, (x, y, yaw))` = observation dict; `units()/rad()` conversions.
2. **Chunk probe** (`OLD/probe.py`): `load`, `batchify`, `chunk(policy, batch, noise)` with `noises(n, seed)`.
3. Standard probes (each a small script under `agents/<you>/`, output saved as npz/json):
   - **κ (servo gain)**: arm frozen at an IK hover, cube moved ±1.5/±3 cm tangential/radial; κ = chunk-end command
     change / IK-required change (`kappa.py`, Cartesian version `kcart.py`). Per-camera by blanking/patching one camera.
   - **Closed-loop-consistent κ**: displace the ARM (state and images consistent), split into state-only and image-only
     parts (`cldec.py`).
   - **Hold / drift**: centred hover chunk-end vs start; first chunk from REST (aim).
   - **Edit equivalence**: function-preserving half reproduces the original to float tolerance (`OLD/check_scale.py`).
   - **Jitter**: within-chunk second difference on recorded observations (`OLD/p1.py…p5.py`, `OLD/jitter_probe.py`).
   - **Activation capture** at a layer for readouts/fits (S04, S05, S07).
4. Script header: coordinate frame, sign convention of every quantity, noise seeds, base variant.
5. Compare against the same probe on the parent variant in the same run.

## Verification
- Sanity: the probe on the original model reproduces a known closed-loop fact (e.g. pan κ ≈ 0.57 for the export).
- Re-render determinism: re-renders of research runs were bit-identical (doc agents' side-by-side videos).

## Expected effect & cost
- Seconds to ~1 min per question vs 18 s simulated / 31–95 s wall per rollout episode (median 50 s).
- p1–p5 found the jitter source in ~1 min each; `scale_axis` was verified before any sim run.
- Every proven edit in the library was verified with these probes first.

## Failure modes / interactions with other strategies
- Too few samples: 4–8 frames at ±10 mm cannot resolve weak effects (Haiku children). Use full suites' worth of
  states, grouped CV.
- Wrong axis / sign: a probe along the vertical axis instead of range, and a reversed flow sign, gave confident wrong
  conclusions. State conventions in the header.
- Open-loop ≠ closed-loop: cube-move κ doubled while closed-loop radial correction stayed ≈ 0 at a yawed pose (S04);
  shadow's correction fit at IK poses double-corrected closed loop (S14). Include re-rendered closed-loop states.
- The renderer must match the deployment condition (stale camera copy, S07).

## Evidence log
- 2026-11-05, SmolVLA, p1–p5 offline probes: jitter is within-chunk; pad-zero d² ~5× lower. Evidence: OLD/probe.py, OLD/p1.py…p5.py; lead b078391b 22:54–23:19Z.
- 2026-11-06/07, servo κ and CL κ tables. Evidence: OLD/agents/servo/REPORT.md §1; kappa.py, kcart.py, cldec.py.
- 2026-11-05, Haiku children's 4–8-frame probes: 0/18 accepted. Evidence: CODEX/sparse-resume-20261105/haiku/parent_review.json.
- Eval cost: 4,289 logged episodes, median 50 s wall. Evidence: scratchpad lead.md §6 (secondary).
