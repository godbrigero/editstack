# S15 — Gated "keep searching until the target is seen" circuit (seek)

Status: partial
Last updated: 2026-11-10

Fixes the reproduced failure strongly and is neutral elsewhere on 2 seeds; never qualified (≥3 seeds) and does not fix
the real far-reach shortfall.

## When to use
- The policy stops making progress when the target leaves a camera's view: chunk pan move ≈ 0.15–0.22 × bearing error
  when the cube is hidden, 0.40 when visible; it descends anyway and closes far away.
- A real-world failure of this shape, reproduced in sim with S10 knobs (first-chunk under-turn).

## Preconditions & confirming probes
1. Reproduction set where the baseline fails (cf3p25 near with chunk-1 pan × 0: REACH 10/24).
2. Readouts at L15 MLP input (2,480 rendered [measured] scenes incl. closed-loop states, ±35 u pan perturbations, IK
   hovers, REST; ridge λ = 10):
   - g3 = "cube not (fully) in the wrist view, tip > 3.5 cm, not REST": target clip((0.85 − pv)/0.5)·ramp(h; 3.5 → 5 cm),
     pv = min(corner fraction, yellow px/(0.7·px_ref(h))) — pixels, not geometry (the open jaw can hide the cube).
     CV: visible 0.02 ± 0.13, hidden 0.88, low 0.01, REST 0.00.
   - bw = bearing error = IK pan for the cube − state pan (units); CV rmse 5.4 u, slope 0.92 on hidden scenes (comes from
     the front camera when the wrist is blind).

## Procedure (`OLD/agents/seek/seek.py seek`)
dP [pan units/chunk] = β·bw·ramp(g3; 0.3 → 0.6), β = 0.7.
Two hinge units (S05): j0 gate = k(g3 − 0.3·one), up = bw, down = +D·q/(kW); j1 gate = k(g3 − 0.6·one), up = bw,
down = −D·q/(kW); W = 0.3, k = 8/W, q = −β/σ_pan, D = pan-only direction. Units 213/1311 (least used on cf3p25).

## Verification
- Offline: other joints' direct effect 0.000 (`leak.py`); Δpan visible 0.01–0.27 u, REST 0.02–0.09 u.
- Closed loop: β 0.7 ≈ deadbeat chunk 2 (rf ep0 pan −9.5 → −26.4 → −30.9, needed −31.8); β 1.2 overshoots 3–4 cm.

## Expected effect & cost
- sk5p vs cf3p25: repro near REACH 10/24 → 22/24; rf repro 8/16 → 8/16 REACH (PICK 5 → 7); ID 39/40 → 38/40 REACH,
  PICK 34 → 35/40; near 24/24 = 24/24 (2 seeds each).
- Cost: ~3 h 20 min agent, ~528 episodes.

## Failure modes / interactions with other strategies
- Corner-only gate (sk09/sk12) misses jaw-occluded cubes.
- A reach-seek term (γ·rr·ramp·JRAD) didn't change REACH beyond noise: the far shortfall happens while the cube IS
  visible (wrist flex at its limit).
- Fit on post-map (S07) activations; refit if upstream maps change.

## Evidence log
- 2026-11-09, SmolVLA cf3p25 [measured cameras], `sk5p`: numbers above. Evidence: OLD/agents/seek/REPORT.md §1–4; OLD/agents/seek/ed/sk5p; OLD/candidates.json "sk5p".
