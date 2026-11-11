# S23 — Whole-layer / sub-space gains inside the network and input-side warps — ANTI-STRATEGY

Status: failed
Last updated: 2026-11-10

## When to use
Don't use these to raise a specific output gain. Inside the network the task code is shared and low-dimensional;
scaling it moves the wrong outputs and the mean. Use the read-out (S04) or a gated circuit (S05).

## Preconditions & confirming probes
Before any internal gain: check whether the target direction is separable from other outputs at that site (principal
cosines between the sub-spaces). At the expert's o_proj outputs, radial and tangential cube codes had top-3 cosines 0.9+.

## Procedure
(What was done:)
- whole-layer `o_proj` scale (L3 ×2, L9 ×2, L13 ×2, L1 ×2, all odd ×1.3);
- o_proj sub-space gain W ← (I + (g−1)UUᵀ)W (selective/unselective, ranks 3–4);
- SigLIP position-embedding column warp (compress perceived lateral position);
- connector projection / full-rank off→on maps to remove the shadow;
- `v_proj` seed-noise damping (hybrid c2_vdamp14).

## Verification
Offline κ (S09) answered each in seconds.

## Expected effect & cost
- L3 ×2: radial κ DOWN (0.13 → 0.10); L9/L13 ×2 raised pan (1.20/1.08) not radial; all odd ×1.3: κ_r 0.12.
- Selective sub-spaces: do nothing (κ_r 0.13–0.14); all layers r4 g1.5: pan ×1.8, radial ×1.7 with 2–8 cm drift.
- PE warp (kk 1.3/1.7/2.5): chunk-1 pan at x = 0 moved −7/−15/−24 u (asymmetric shift, not compression).
- Shadow maps: < 10 % / leaves 78–89 %.
- vdamp14: REACH +1 (< 1 SE), held-out loss +9.7 %, not applied.

## Failure modes / interactions with other strategies
- Whole-layer scaling shifts the mean → first chunk changes (≤ 1.7–2.9 cm).

## Evidence log
- 2026-11-07, SmolVLA pan17 base. Evidence: OLD/agents/servo/REPORT.md §3, §5; OLD/agents/pancost/REPORT.md §6; OLD/agents/shadow/REPORT.md §4; HYB/EDIT_LEDGER.md "c2_vdamp14".
