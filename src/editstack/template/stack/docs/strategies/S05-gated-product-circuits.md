# S05 — Gated circuits in spare SwiGLU units (exact product pair + soft hinges)

Status: proven
Last updated: 2026-11-10

## When to use
- You need a NONLINEAR correction: a gain that differs in-band vs out-of-band, a term that must be exactly 0 while
  holding still, a correction that should fire only in one phase (open gripper AND high; cube not visible).
- The quantities the correction needs are linearly readable (R² ≳ 0.95) at the input of a late MLP.
- A linear edit (S03/S04) helps one region and costs another.

## Preconditions & confirming probes
1. **Site**: the last expert MLP (SmolVLA `model.vlm_with_expert.lm_expert.layers.15.mlp`). Its input (post-RMSNorm
   residual) linearly encodes state, planned actions, cube x, gripper state, tip height, jaw/cube angle (MODEL_MAP), and
   only the final RMSNorm + `action_out_proj` follow, so the output direction can be solved exactly.
   The MLP is h_j = silu(g_j·x)·(u_j·x), y += down[:, j]·h_j, with NO biases.
2. **Readouts** (ridge, λ = 10, no intercept, 5-fold CV) from captured MLP inputs X: d = planned − state on the axis,
   s = state, xc = cube x, gate variables, and **`one`** (w·x ≈ 1, rmse 0.006) which supplies constants/thresholds in a
   bias-free MLP. Capture design (`OLD/agents/approach/capture.py`): 635–735 scenes = REST ± jitter, IK hovers at
   2/5/8/12 cm, mid-reach, and re-rendered closed-loop chunk-boundary states; 10 Euler steps × 11 tokens each.
   Past CV rmse: d 1.05–1.59 u (sd 5–8.5), s 1.17–1.44 u (sd 18), xc 1.37–1.48 cm (sd 8.8).
3. **Spare units**: importance_j = mean|silu(g_j·x)(u_j·x)| × ‖down[:, j]‖ over the captures; take the bottom
   0.2–0.5 % (past 0.0057–0.008 vs median 0.047). Overwriting them must leave unrelated chunks unchanged (≤ 0.1 units).
   Use `ops.free_units(ctx.stack, W, [(gate,0),(up,0),(down,1)], k, score=importance)` and claim every unit.
4. **Noise coupling check**: a readout that also reads the noisy action x_t feeds back over the 10 Euler steps
   (grasp: 50–2000-unit kicks, even at REST). Project readouts off the top-8 singular directions of dh/dx_t
   (finite differences) or orthogonalise against them.

## Procedure
1. **Output direction** for "change joint j's velocity by +1 (normalised), nothing else":
   D = r · pinv(W_out[:6] ⊙ γ) e_j, γ = final RMSNorm weight, r ≈ 3.3 (the norm's typical rms; 3.3–3.6, sd 10 %).
   A velocity change dv at all 10 Euler steps moves the chunk by −dv (normalised), i.e. by −dv·σ_j raw units.
   (`OLD/agents/approach/gpg.py pan_direction`, `OLD/agents/grasp/grs.py joint_direction`.)
2. **Exact product** (silu(z) − silu(−z) = z):
   unit 1: gate = a·w1, up = b·w2, down = D·s/(ab); unit 2: gate = −a·w1, up = −b·w2, down = same
   ⇒ adds s·(w1·x)(w2·x)·D exactly. `ops.exact_product_pair(W, ctx, gate, up, down, pair, w1, w2, D*s, purpose)`.
   With w2 = `one`, it is an exact linear unit.
3. **Soft hinge**: gate = k(w·x − t), up = w2, down = D·c/k ⇒ c·silu(k z)/k·(w2·x) ≈ c·relu(z)(w2·x); silu dip ≤ 0.28/k.
   **Ramp** ρ(z; z0→z1) = [h(z−z0) − h(z−z1)]/(z1−z0), k = 8/(z1−z0) (2 units). Two-sided |g| gate: 4 units.
   Write each unit with `ops.swiglu_unit(W, ctx, gate, up, down, idx, g_in, u_in, d_out, purpose)`.
4. **Gate design — make it exactly 0 where the model already works**:
   - multiply by d = planned − state (0 while holding → no drift possible);
   - ramp on an out-of-band variable (|cube x| 6.5 → 9.5 cm);
   - AND-gates as one combined variable, e.g. z = (g − 85) + 8.5·(h − 4.5) (gripper open AND tip ≥ 4.5 cm).
5. Worked example `gated_gain` (6 units): dP = −c·d·(1 − ρ(|xc|)), c = 0.41 = 1 − 1/1.7:
   2 exact units for −c·d, 4 hinge units (±xc, thresholds g0, g1, up = d) for +c·d·ρ. Velocity coefficient s = c/σ_pan.
6. Save readouts and the chosen units to `edits/<v>/data/`; docstring: gate, magnitude, units, exactness argument.

## Verification
- Rung a offline (`OLD/agents/approach/verify.py`, `OLD/agents/circuit/verify.py`): (i) unrelated chunks unchanged
  (straight-ahead cubes ≤ 0.1 units; hold test ≤ 0.05–0.33 cm); (ii) the gated quantity changes by the designed amount
  in the gated region; (iii) **realised vs nominal gain** — past 0.8× (circuit reach) and ~0.5× (gated hold) because of
  the norm-scale approximation and bf16 rounding (~0.4 %); calibrate before the sim.
- `editstack apply`: only the claimed rows/columns changed; the claim list has no overlaps.
- Rungs b–d; check BOTH the gated-on set and the in-band set.

## Expected effect & cost
- p17gs (gated pan gain + gated sp6 servo, 10 units): near REACH 35/36 (pan17 31/36), ID REACH 38/40, ID PICK 33/40;
  in-band chunk-1 aim 1.53 → 1.03. p17gsh (+ 6-unit gated hold pull): ID REACH 40/40, PICK 34/40.
- circuit c11/db10 (reach p²−s²): radial error −2.0 → ≈0 cm, but near PICK 19/24 vs 20/24 (null; the binding
  constraint was grasp). A correct mechanism is not a score gain: check the failure taxonomy first.
- Hybrid ENDEDIT: four later dead-zone circuits added 0 on the main suite (m2 = m7 = 60/80).
- Cost: one capture run, minutes to fit, 2–6 units per term.

## Failure modes / interactions with other strategies
- **Unit collisions**: approach's servo/hold and grasp's roll both used L15 779/1893; caught on paper by approach's
  REPORT (not by a claims list), reassigned to 1263/248. The shared MODEL_MAP registry itself wrongly listed 1893/1701 as
  free. Use `ctx.claim` and derive claims from code, never from a hand-kept list.
- Overshoot is fatal, undershoot is tolerated; keep gains small and asymmetric where possible.
- Readouts must be refit on the base the circuit is stacked on (readout_L15_p17 vs _exp vs _r1a).
- Softclip on pan coordinate (in-band fine, out-of-band −9 u constant) and state-only hold compensation (phase-dependent,
  over-pushes in descent) failed; gate on the task variable instead.
- In a training loop, small bf16 up/down weights drift under Adam and inputs drift ~38 %/segment (S18).

## Evidence log
- 2026-11-07, SmolVLA (pan17 base), circuit `c11`, `db10`: reach fixed, PICK null. Evidence: OLD/agents/circuit/REPORT.md §3–5; OLD/agents/circuit/circ.py.
- 2026-11-07, `p17gs`: near R35/36 P29/36, ID R38/40 P33/40. Evidence: OLD/agents/approach/REPORT.md; OLD/agents/approach/gpg.py; OLD/agents/approach/ed/p17gs.
- 2026-11-07, `p17gsh`: ID R40/40 P34/40, near R35/36. Evidence: OLD/agents/approach/REPORT.md "p17gsh".
- 2026-11-07, neuron collision 779/1893 found in approach's REPORT; registry error. Evidence: OLD/agents/approach/REPORT.md; lead b078391b 2026-11-07T23:46Z; scratchpad review/REVIEW.md §1.1.
- 2026-11-09, hybrid ENDEDIT m2 → m7 unchanged at 60/80. Evidence: HYB/EDIT_LEDGER.md "ENDEDIT".
