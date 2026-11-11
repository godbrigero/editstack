# S06 — Gate an existing fix to where it is needed (remove its in-band cost)

Status: proven
Last updated: 2026-11-10

## When to use
- A global edit (S03 scale, S04 servo) fixes the out-of-band set but costs in-band (pan×1.7: ID PICK 73 → 63/80).
- Per-region gain shows over-correction in band: pan×1.7 chunk-1 aim 1.46–1.56× needed in band, 1.05–1.08 out.
- sp6's radial gain overshoots in band (ep6 "l", +1.3…+2 cm; in-band drift +0.69…+1.00 cm vs pan17's +0.18…+0.76).

## Preconditions & confirming probes
1. **Diagnose the cost first** (pancost pattern): regress chunk-1 move on needed move per region and seed
   (`OLD/agents/pancost/aim1.py`), relate the residual error at close to pick rate (0.96/0.86/0.70/0.25 for
   <0.3/0.3–0.6/0.6–1.0/>1 cm), and predict the gap from the error distribution (71.4 vs 64.5 predicted; 73 vs 63 observed).
2. Rule out side effects of the global edit (pancost: noise std 1/k ≤ 0.06 units; other joints reading the state ≤ 0.12).
3. A band variable must be linearly readable at the circuit site (xc: CV rmse 1.4 cm, sd 8.8 cm — coarse but enough
   for a 3 cm ramp).
4. For a read-out edit you want to gate: check it factorises. sp6's ΔW,Δb is exactly rank 1: change = q·(w·h + β),
   so the scalar t = w·h + β can be re-read at the L15 MLP input (R² 0.998) and multiplied by the gate.

## Procedure
1. Keep the global edit in the chain (it provides the out-of-band behaviour exactly) and add a circuit (S05) that
   subtracts its effect in band:
   **dP = −c·d·(1 − ρ(|xc|; 6.5 → 9.5 cm))**, c = 1 − 1/k (0.41 for k = 1.7), d = planned − state.
   In band every pan move gets the original gain; out of band it is pan×1.7 exactly; 0 while holding.
   Code: `gpg.gated_gain(sd, ASTD, c=0.41, g0=0.065, g1=0.095, readout=RO)` (6 units).
2. For a read-out servo: don't apply it globally; apply its scalar through the gate:
   dv[rows 1..3] = q·t·ρ(|xc|), t from readout 't6' (`gpg.gated_servo`, 4 hinge units, output by pinv).
3. If a residual in-band drift remains (pan17 lacked the export's slight inward hold pull: export − pan17 = −0.043·s
   u/chunk), add a gated state term: dP = −γ·s·(1 − ρ) (`gpg.gated_hold`, γ = 0.08 nominal, realised ≈ 0.5×).
4. Readouts are fit on the chain's base (pan17), never reused from another base.

## Verification
- Rung a: per-region chunk-1 aim (in band → ~1.0; out of band unchanged), in-band radial κ/drift equal to the
  un-servoed base and out-of-band equal to the servoed one (approach: κ_r 0.68/0.34/0.60/0.49/0.21 vs sp6
  0.69/0.35/0.62/0.48/0.23).
- Rungs b–d on BOTH sets, ≥3 seeds.

## Expected effect & cost
- p17gs vs pan17: in-band chunk-1 aim 1.53 → 1.03 (tangential rms 6.3 → 2.0 u); near REACH 31 → 35/36; ID REACH 37 → 38/40.
- p17gsh: ID REACH 40/40 (export 40/40) with near 35/36; ID PICK 34/40.
- Cost: 6–16 units at L15; a capture + readout fit per base.

## Failure modes / interactions with other strategies
- Gating on the pan coordinate instead of the cube (softclip) fails: the band edge in pan depends on y and overlaps the
  near set; restoring the out-of-band aim then needs slope > 1 (hover κ > 1.5).
- Out-of-band boost on the un-scaled base + state-only hold compensation (xbh2) failed (near 4/8): the base's hold
  compression is multiplied by the boost. Scale globally (S03) and gate in band, not the reverse.
- Gating a perception-level edit (r1a_xg) fixes aim but not that edit's own in-band error (ID PICK 28/40).

## Evidence log
- 2026-11-07, pancost diagnosis of pan×1.7 ID cost (73/80 → 63/80, aim 1.46–1.56). Evidence: OLD/agents/pancost/REPORT.md §1.
- 2026-11-07, `p17gs`, `p17gsh`, `xbh2`, softclip. Evidence: OLD/agents/approach/REPORT.md; OLD/agents/approach/gpg.py; OLD/agents/approach/radial.py.
- 2026-11-07, sp6 rank-1 factorisation (2.40 vs 1.6e-14), t6 R² 0.998. Evidence: OLD/agents/approach/REPORT.md TL;DR; gpg.sp6_factor.
