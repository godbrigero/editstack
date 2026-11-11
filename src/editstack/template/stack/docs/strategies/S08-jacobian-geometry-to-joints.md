# S08 — Convert task-space errors to joint/action deltas with the kinematic Jacobian

Status: proven
Last updated: 2026-11-10

## When to use
- You measured an error in task space (mm short along the jaw axis, cm radially short) and need the output-space
  vector to feed an exact edit (S02 translation, S04 projection direction, S05 output direction).
- You want a physically meaningful direction to project a fit onto (removes side effects such as vertical response).

## Preconditions & confirming probes
- A forward-kinematics model of the robot (MuJoCo model of the sim), the action ↔ joint-angle unit mapping
  (SmolVLA: ±100 range → rad, then mean/std normalisation), and stored trajectories with geometry (`traj_*.npz`:
  actions, states, geo = cube xyz | tip xyz | jaw xyz).
- Measure the error at the decisive moment from stored trajectories (`OLD/geo.py`): past, the tip closed 6–10 mm
  short along the jaw axis.

## Procedure (`OLD/jac.py`, `OLD/jac2.py`)
1. For each episode, find the close (gripper command opens > 80 then < 50) and take the state there.
2. Convert state units → q (rad), compute tip position by FK, and a numeric Jacobian J = ∂tip/∂u for joints 0..3
   (pan, lift, elbow, wrist-flex) in action units, constraining the gripper orientation (keep the approach axis).
3. Solve Δu = J⁺ Δtip for the wanted move (e.g. +8 mm along the gripper axis) and average over episodes.
   Result: Δ = [0, +1.52, +1.52, −3.11, 0, 0] raw units → S02.
4. Directions for projections: normalised radial direction URAD = (0.62, −0.77, 0.12) on (lift, elbow, wflex)
   (kappa probe; cos 0.85–1.0 with the model's own radial response); JRAD = [0, 4.23, −6.41, 0.59] units per cm at
   5–8 cm hover (seek); reach J = [0, 400, −480, 0] units per metre (circuit, wrist flex saturated at hover).
5. Store Δ / directions in `edits/<v>/data/` with the script path.

## Verification
- Offline: apply Δ in FK at the recorded close states and check the tip moves by the target (mm).
- After the edit: re-measure the geometric error closed loop (the error you meant to remove), not only success.

## Expected effect & cost
- Minutes; turned "closes 8 mm short" into an exact 4-tensor edit in 74 min from project start (with S01, S09).
- Provides every output direction used by S04/S05/S15.

## Failure modes / interactions with other strategies
- Joint limits: wrist flex saturates at hover and at far reach (seek: wrist flex at 99.9 for the far placements);
  the Jacobian then needs the remaining joints, and the edit can't deliver the full move.
- A Jacobian at one posture is a local linearisation; average over the postures where the edit acts.

## Evidence log
- 2026-11-05, SmolVLA step-6000, `jac.py`: Δ = [0, 1.52, 1.52, −3.11, 0, 0] for 8 mm. Evidence: OLD/jac.py, OLD/geo.py; lead b078391b 2026-11-05T22:52–22:53Z; OLD/ed/pad0_deep8/EDIT.py.
- 2026-11-07, servo URAD; circuit J; 2026-11-09 seek JRAD. Evidence: OLD/agents/servo/sv.py; OLD/agents/circuit/jacr.py; OLD/agents/seek/seek.py.
