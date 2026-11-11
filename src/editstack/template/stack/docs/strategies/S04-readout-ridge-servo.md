# S04 — Closed-form servo gain at the read-out (rank-1 ridge on recorded activations)

Status: proven
Last updated: 2026-11-10

## When to use
- A feedback correction exists but is too weak: per-chunk κ on some direction ≪ 1 (past: radial κ ≈ 0.13–0.15
  per chunk, needed ~0.5–1) while the direction of the response is right (cos 0.85–1.0 with the IK direction).
- The arm "stays short" through hover and descent (ep5/ep11: 3.5–5 cm radially short until the grasp).
- The read-out input separates the signal from the hold: principal cosines between cube-induced and state-induced
  variation of the read-out input were 0.32/0.18/0.05 (near-orthogonal), so the image part can be amplified alone.

## Preconditions & confirming probes
1. Closed-loop-consistent κ decomposition (`OLD/agents/servo/cldec.py`): displace the ARM (state + both images
   consistent), split into state-only and image-only parts. Past radial: total +0.01…+0.07, state-only −0.03…−0.09
   (the state pathway over-follows by 3–9 % and cancels the image term), image-only +0.07…+0.14.
2. Where the image signal enters (KV patching, `patch.py`): odd cross-attention layers only; state token 0.01–0.03.
3. Do NOT start with layer/sub-space gains inside the network (S23): the cube code there is shared by pan and radial.
4. Decide what must stay unchanged: first chunk from REST, consistent (centred) scenes at all heights, and (if
   relevant) descent/vertical response.

## Procedure (sp6; `OLD/agents/servo/sv.py amp_servo`, statistics from `collect3.py`, `collect4.py`)
h = input of `action_out_proj` (720-d, all 50 tokens × 10 Euler steps), R = W_out[rows], rows = (lift, elbow, wrist-flex).
1. **Record paired scenes with the current best model** (S09 renderer, 4 fixed noises, deltas noise-averaged):
   8 hover bases × 3 heights (10/6/3 cm), varied yaw. At each base: cube moved ±2 cm tangential and radial with the arm
   fixed → dc; ARM moved ±2 cm with the cube fixed, state + images consistent → da, and its parts: state-only ds
   (centred images, displaced state) and image-only dv (displaced images, centred state).
2. **Second moments**: Gcc = ⟨dc dcᵀ⟩, Gaa = ⟨da daᵀ⟩, Cva = ⟨dv daᵀ⟩, Csa = ⟨ds daᵀ⟩; Gq = Gram of h-deltas for cube
   moves with the arm at REST; M0 = ⟨[h,1][h,1]ᵀ⟩ over consistent scenes (centred hover at 3 heights, grasp pose, REST).
3. **Targets** (extra output wanted): cube moves (g−1)·R dc; arm moves (g−1)·R dv + (gs−1)·R ds (image part × g,
   state part × gs < 1 to undo the over-follow); REST cube moves → 0 (weight λq = 30); consistent scenes → 0 (λ0 = 10).
4. **Solve** X = [ΔW Δb] (3 × 721): X·A = T with
   A = aug(Gcc) + aug(Gaa) + λq·aug(Gq) + λ0·M0 + 1e-4·tr(A)/721·I,
   T = [R((g−1)(Gcc + Cva) + (gs−1)Csa), 0].
5. **Project onto the physical direction**: T ← QQᵀT with Q the orthonormal basis of URAD = (0.62, −0.77, 0.12) (the
   IK horizontal-radial joint direction, normalised units). This makes the edit **exactly rank 1** (singular values
   2.40 vs 1.6e-14) and removes the vertical side effect.
6. Apply: `ops.add_delta(W, "model.action_out_proj.weight", D_full, max_rank=1)` where D_full is zero except rows
   1..3 = ΔW; `W["model.action_out_proj.bias"][1:4] += Δb`. Save X to `edits/<v>/data/` (the fit is ~1 s on CPU).
   `fit.fit_delta` covers the simpler case (one set of rows with absolute targets); for "delta of delta" targets with
   protected scenes, write the weighted Gram solve above.
sp6 settings: g = 6, gs = 0.7, λ0 = 10, λq = 30, proj = URAD.

Hybrid variant (`c2_tipservo_v2`): U = 4×720 ridge probe on the read-out input with target = per-token ramped
((k+1)/50) constant-height IK correction, target 0 at home poses, post-grasp/lift frames and demo frames with their own
cube; U orthogonalised against the top-K noisy-action coupling directions; `W[j] −= g_j U_j`, `b[j] −= g_j(ym_j − U_j·μ)`,
g = 0.8/1.5/1.5/1.5 (pan, lift, elbow, wflex).

## Verification
- Rung a: offline evaluator (`OLD/agents/servo/ev.py`): cube-move κ (kt/kr), closed-loop κ (CL cr), vertical response
  kz, hold drift and REST first-chunk drift. sp6: CL radial 0.12–0.14 → 0.44; kz 0.15 → 0.20; drift ≤ 0.6 cm.
  Reject if kz grows (w2g6: 0.70) or REST drift > ~1 cm.
- Also check the noise coupling: the read-out must not read the noisy action x_t (else runaway over Euler steps).
- Rungs b–d, ≥3 near seeds; report the per-episode string for the episodes the fix targets.

## Expected effect & cost
- sp6: near REACH 31/36 → 33/36 (~1 SE), the targeted radial failures ep5/ep11 1/6 → 5/6; ID REACH 36/40 (pan17 37/39),
  ID PICK −4 on seed 0 (lift/elbow jitter +50%). Became best-thus-far 2026-11-06 23:13 local.
- Hybrid c2_tipservo_v2: main REACH 5 → 27/80 (+27.5 ± 5.6 pp), PICK 4 → 20; held-out loss +6.4%.
  On the finished control checkpoint: m1 (pad-zero) 44/80 → **m2 (+ this servo) 60/80**, the full ENDEDIT score.
- Cost: one capture run (minutes–1 h of renders), seconds to fit.

## Failure modes / interactions with other strategies
- Amplifying CUBE moves only (wi5, w2g6) fails closed loop: in the loop the arm moves, and arm-move images differ from
  cube-move images; it also amplified the vertical response ×4 (kz 0.70) → arm hovered 12 cm for a whole episode.
- Too much gain (sp8, CL 0.50) adds jitter (lift/elbow 0.64/0.79) and overshoots ("overshoot is fatal, undershoot is not").
- Active in band → in-band overshoot (ep6 "l"); gate it out of band (S06: `gated_servo`, rank-1 factor re-read at L15).
- Double correction: shadow's radial fix on top of the gated servo pulled 1.2–1.8 cm short (S14).
- In a training loop it washes out (S18).
- Hybrid `c1_panrel` (rank-1 P-controller on the pan row): held-out loss +164 %, PICK 4 → 1. A read-out servo must follow
  the per-token ramp, include the depth joints, know the phase and use a moderate gain.

## Evidence log
- 2026-11-06/07, SmolVLA step-6000 (pan17 base), `sp6`: near R33/36 P30/36, ID R36/40 P28/40; ep5/ep11 1/6 → 5/6. Evidence: OLD/agents/servo/REPORT.md §3–4; OLD/agents/servo/ed/sp6/EDIT.py; OLD/export/BEST_THUS_FAR.md.
- 2026-11-07, `sph6` (height-gated): near R32/36, ID R37/40 — a wash vs sp6. Evidence: OLD/agents/servo/REPORT.md §4.
- 2026-11-07, hybrid step 1000, `c2_tipservo_v2`: REACH 5 → 27/80. Evidence: HYB/EDIT_LEDGER.md "c2_tipservo_v2".
- 2026-11-07, hybrid step 500, `c1_panrel`: reverted (loss +164 %). Evidence: HYB/EDIT_LEDGER.md "c1_panrel".
- 2026-11-09, hybrid control/6000, ENDEDIT m2: 60/80 (= final m7). Evidence: HYB/EDIT_LEDGER.md "ENDEDIT"; HYB/agents/endedit/chain.log.
