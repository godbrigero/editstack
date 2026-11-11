# S16 — Perception-level rank-1 rewrite of an attention output (read the true quantity)

Status: partial
Last updated: 2026-11-10

Reach was fixed in closed loop, but it added in-band error and gave no pick gain; dropped from the chain. Prefer
read-out-side edits (S04–S06) unless the read layer is the only place the quantity exists.

## When to use
- The model reads the wrong function of a perceived quantity: reach set from forward distance |y|, not radius r
  (perceived ŷ ≈ |y| beyond the band; x = ±0.12, y = −0.24 → ŷ 0.229/0.240 vs r 0.268).
- The shortfall t = r − ŷ is linearly decodable from the reading layers' attention outputs (scene-level CV R² 0.86–0.92).

## Preconditions & confirming probes
1. Causal location by K/V patching (latent `patch1.py`, `patch2.py`): reach read at cross-attention L3 (0.43–0.75) and
   L5 (0.24–0.31); even layers 0.00.
2. Gain of the reading layers: injecting s·u_L moves ŷ by 0.3·s (one layer), 0.65·s (L1+3+5+7).

## Procedure (`OLD/agents/latent/latedit.py apply`)
For L ∈ {1, 3, 5, 7}: O_L ← O_L + κ·(O_L u_L) w_Lᵀ (rank-1 on `self_attn.o_proj`, 720 × 960), i.e. z_L ← z_L + κ u_L (w_L·z_L).
- u_L: slope of z_L on ŷ over in-band scenes, regressing out x and yaw².
- w_L: no-intercept ridge readout of z_L with target s = t*/0.65, t* = (r − ŷ_smooth)·ramp(|x|; 0.05 → 0.09), 0 at hover;
  613 REST + 80 hover scenes, λ = 3.
- editstack: `ops.rank1(W, o_proj, u=O_L @ u_L, v=w_L, alpha=κ)`.

## Verification
- Offline first chunk from REST (unseen noise): mean |ŷ − r| on near placements 1.88 → 0.72 cm; ID scenes moved ≤ 2.3 u.
- Closed loop: radial error at close and in-band side error, not only PICK.

## Expected effect & cost
- r1a: near radial error at close ep5/ep11 −3.1/−4.1 → +0.5/0.0 cm; near PICK 9/12 vs 10/12 (2 seeds); later REACH 23/24.
- ID: REACH 34/40, PICK 29/40; in-band side rms 0.90 cm vs pan17 0.67. r1a_xg (+ gated gain): near 31/36 after the 3rd seed.

## Failure modes / interactions with other strategies
- Adds in-band error that later gating cannot remove (r1a_xg ID PICK 28/40); approach dropped it.
- Rank-2 variant r2a (keep aim angle) 7/12: worse.
- Pan magnitude shrinks 1–8 units as a side effect (the model aims at atan(x/ŷ)).

## Evidence log
- 2026-11-07, SmolVLA pan17 base, `r1a`, `r2a`. Evidence: OLD/agents/latent/REPORT.md §1–2; OLD/agents/latent/latedit.py.
- 2026-11-07, r1a ID check, r1a_xg. Evidence: OLD/agents/approach/REPORT.md TL;DR and results table; OLD/export/BEST_THUS_FAR.md.
