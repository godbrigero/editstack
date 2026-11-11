# S07 — Fold a paired-render linear map upstream to survive an input shift

Status: proven
Last updated: 2026-11-10

Key property: the map sends new-condition activations back to the training-condition activations, so **every
downstream edit (read-out servos, gated circuits and their readouts) keeps working without a refit**. Two days of
circuits carried over unchanged to the measured cameras.

## When to use
- Everything got worse after a sensor/camera/preprocessing/rendering change (p17gshr under measured cameras: ID REACH
  40/40 → 6/40, PICK 4/40; near 15/36).
- The real system's inputs differ from the training/sim inputs (real cameras: front FOV 42.6° vs sim 64°, wrist 59.3° vs
  80.7°; the RealSense also delivered its IR stream).
- You can render or record the SAME physical states under both conditions.

## Preconditions & confirming probes
1. **Harness first** (HARNESS_AUDIT.md): the sim must actually render the new condition. Camfix found the eval was
   loading a stale `repo/` copy with the old cameras.
2. **Per-layer shift size**: on paired renders, |o_new − o_old|/|o_old| per layer at the candidate site. Past: expert
   attention outputs moved 30–60 % at the odd (cross-attention) layers.
3. **Linearity check**: held-out "fraction of the shift left" after a ridge map, per layer (5 folds). Proceed only where
   it is well below 1. Counter-example: the arm-shadow change left 78–89 % after a full-rank connector map — the
   information was absent, not transformed (S14).

## Procedure (`OLD/agents/camfix/cap.py`, `fit2.py`, `cf.py apply_maps`)
1. **Scenes** (S09 renderer): REST ± jitter (¼), IK hovers at 2/4/6/9/12 cm with ±1.5 cm offsets (⅓), mid-reach (⅛),
   and re-rendered closed-loop chunk-boundary states from stored runs (⅓). cf1: 500 scenes. cf2: 980 scenes with 50 %
   edge placements |x| 0.07–0.16, 15 % near the real placements. cf3: + misaligned low hovers.
2. **Teacher-forced paired capture** for each scene and fixed noise:
   run 1 (training view) records the input of every expert `o_proj` (16 layers × 10 Euler steps, 6 tokens);
   run 2 (new view) records the same inputs but REPLACES them by run 1's at every layer, so each layer's new-condition
   activation is computed from the training upstream state (clean per-layer pairs, no error compounding);
   run 3 (new view, free) gives the uncorrected baseline error.
3. **Per-layer map, ridge shrunk to the identity** (not to zero): with X = new, Y = training (rows = samples),
   A_L = (XᵀX + λI)⁻¹(XᵀY + λI), λ = λ'·n_rows/1000, λ' chosen per layer by 5-fold CV from {0.3, 1, 3, 10}
   on "fraction of the camera-induced difference left" ‖Y − XA‖²/‖Y − X‖².
   (`fit.paired_map` shrinks to zero with an intercept; for this use compute the identity-shrunk solve yourself.)
4. **Fold**: o_proj computes y = W·o; we want W·(A_Lᵀ o_new): `ops.fold_input_map(W, f"...layers.{L}.self_attn.o_proj.weight", A_L.T)`
   for all 16 layers (`cf.apply_maps` does W ← W·(I + α(A − I))ᵀ, α = 1).
5. Optionally restore a gain the map does not fully recover with a small gated circuit (cf3p25: `cf.low_pan_gain`,
   β 0.25, height-gated pan servo, κ_tan at 3 cm 0.64–0.69 vs 0.80 under training cameras).

## Verification
- Rung a: per-layer held-out fraction left; offline chunk error vs the training-view chunk on held-out scenes AND
  held-out placements (camfix's CV was by scene, not placement — partly optimistic).
- Rungs b–d under the NEW condition (mark runs with the condition; rank only same-condition runs).
- Real hardware only with a written expected result (rung e).

## Expected effect & cost
- Under measured cameras (bench totals from OLD/candidates.json): p17gshr ID REACH 6/40 → cf1 57/60 (P49), cf2 58/60
  (P48), cf3p25 59/60 (P51); near cf1 33/36, cf2 35/36 (P27), cf3p25 34/36 (P29). cf1 → cf2 → cf3 differences are
  within noise; the 500-scene map already did the work.
- cf1t5/cf3p25t5 (with S02 real offset) picked on the real arm (small n).
- Cost: ~4 h agent time including captures (~1,173 episodes of evals); capture ≈ 3 forward passes per scene.

## Failure modes / interactions with other strategies
- Overfitting: closed-loop states came from the same evaluation placements; CV by scene. Hold out placements.
- Fails when the new condition removes information (shadows) rather than transforming it (S14).
- Must sit UPSTREAM of the edits it protects; downstream readouts stay valid only because activations are mapped back.
- Seek (S15) and later circuits were fit on cf3p25 (post-map) activations.

## Evidence log
- 2026-11-09, SmolVLA p17gshr chain, measured cameras: ID R6/40 P4/40, near 15/36. Evidence: lead b078391b (2026-11-09); scratchpad review/REVIEW.md §1.1.
- 2026-11-09, `cf1`, `cf2`, `cf3p25`: see numbers above. Evidence: OLD/agents/camfix/ed/{cf1,cf2,cf3p25}/EDIT.py; OLD/agents/camfix/{cap.py,fit.py,fit2.py,cf.py}; OLD/candidates.json; OLD/export/BEST_THUS_FAR.md (cf2 2026-11-09 20:56).
- 2026-11-10, audit: the camera mismatch did not waste ~2 days; circuits carried over through the map. Evidence: scratchpad review/REVIEW.md §1.1.
