# S14 — Gated correction for dependence on a sim-only artifact (arm shadow)

Status: partial
Last updated: 2026-11-10

Works on the base it was fit on (sp6); double-corrects on a chain that already compensates (p17gshr). Measure the
dependence first; edit only if the current chain still shows a systematic bias.

## When to use
- The deployment lacks something the sim has (real table: no arm shadow) and performance drops when you remove it
  in sim: p17gshr near REACH 35 → 30/36, PICK 32 → 26/36; ID REACH 40 → 34/40, PICK 34 → 27/40.
- The drop has a SYSTEMATIC signed component (sp6 shadow-off: chunks end +0.3–0.6 cm long; sideways κ −15–20 %).

## Preconditions & confirming probes
1. Paired renders with/without the artifact, same state and noise (`OLD/agents/shadow/sscene.py`: shadow off =
   `light_castshadow = 0`; arm-only; cube-only/table-only masks).
2. Signed chunk-end shift OFF − ON per phase (`bias.py`) and κ with/without (`kappa_sh.py`) ON THE CURRENT CHAIN, in
   closed loop as well as at IK poses. p17gshr closed-loop shadow-off runs showed no systematic long bias → don't edit.
3. Is the artifact information transformed or absent? Connector projection removed < 10 %; full-rank off→on map left
   78–89 %; per-scene v_proj map ≥ 90 % → absent: S07 does not apply; only gated corrections of the systematic part.
4. A clean "artifact absent" gate: ridge at L15 kv-group 0 pooled values ON 0.00 ± 0.20 / OFF 0.88 ± 0.20; with
   "AND tip ≤ 5 cm" at the L15 MLP input: OFF h ≤ 3 cm 0.93–1.03, ON 0.04–0.11.

## Procedure (`OLD/agents/shadow/shed.py`; `ed/sp6_shpg3`)
1. Radial part: rank-1 `lm_expert.layers.15.self_attn.v_proj.weight[0:64, :] += a gᵀ`, g = hover gate readout,
   a = Σ_d c_d·pinv(O_sum)·P(γ ⊙ W_out[d]), c by least squares on finite-difference chunk responses (10 training cubes ×
   5 phases), fit on dims 0–3 so it adds no pan drift. Refit per base.
2. Sideways gain: exact product pair (S05) with gate rows ±w_g (no shadow AND tip ≤ ~5 cm), up rows ±w_d (planned −
   state pan), down = D for v_pan = −0.5·g·d·52.09/σ_pan.

## Verification
- Offline held-out: OFF chunk change vs ON 1.00 → 0.59; edited ON change 0.10. κ_pan OFF 0.52/0.85 → 0.63/1.04
  (ON 0.65/1.00 unchanged).
- Closed loop on BOTH ON and OFF sets, ≥3 seeds each.

## Expected effect & cost
- sp6_shpg3: OFF near REACH 16 → 20/24, PICK 12 → 16; OFF ID REACH 31 → 40/40, PICK 19 → 28/40; ON unchanged-ish
  (near 23/24, 19/24; ID 36/40, 32/40).
- On p17gshr: full edit near OFF PICK 6/12 vs 10/12 (closes 1–2 cm short); pan part only 7/12. Not adopted.
- Cost: the most expensive agent of the project (~9 h, ~463 episodes; shared RAM pool).

## Failure modes / interactions with other strategies
- **Double correction**: p17gshr's gated servo/hold already supplies the radial correction; the shadow radial term on top
  pulled the tip 1.2–1.8 cm short. Ask "does an earlier link already correct this quantity?"
- Correction fit at open-loop IK poses ≠ closed-loop states; refit in closed loop.
- Unconstrained side effect (+0.6–1.0 pan units/chunk in sp6_shg) makes misses one-sided.

## Evidence log
- 2026-11-07, SmolVLA sp6/pan17/p17gshr shadow-off table; sp6_shpg3; p17gshr_sh/_shp. Evidence: OLD/agents/shadow/REPORT.md (headline, §4–6); OLD/agents/shadow/ed/*.
