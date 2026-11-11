# Editing a model by its numbers alone

> **The evidence in these guides comes from a PAST reference project** (a SmolVLA policy on an SO-101 arm, Nov 2026).
> Its numbers, placements, tensor names and neuron indices describe THAT model, not the one in this stack. Use them
> as lessons about method only. Facts about YOUR model come only from `model/` and your own runs in `runs/`.

Read this when the brief is "here are the weights (text dumps, anonymous tensors L1..Ln) and maybe a short
structural note. Fix behaviour X. No backprop." You may get no model code, no names and no training script.
The job is the same as usual: turn "it fails at X" into "quantity Q is wrong by N at place P because of mechanism
M" (ANALYSIS_GUIDE.md), then apply an exact edit. What changes is where each fact comes from: shapes, statistics,
the description, and re-implementing small blocks in numpy.

The archetype is a folder that holds `model_info.md` (9 KB, SmolVLA with every name removed),
`model.txt` (L1–L5, 22 MB), `model_2.txt` (L6–L397, 5.9 GB) and `eval_output.py` (a fake scorer). The task given
was, in short: here are the description and the weight dumps; the policy jitters and fails to pick up the cube; fix
it by editing the weights by hand, no backprop (b078391b 22:40Z Nov 5; Codex parent 04:22Z Nov 5).

Citation key: `b078391b HH:MM` = lead Claude session (UTC, Nov 5 unless stated). `Codex HH:MM` = Codex parent
rollout 2026-11-04T23-22-02. `ID-agents` = the Codex numbers-only identification rollouts 21-28-44 / 21-32-21.
`np` = recomputed with numpy on model.txt for this guide (evidence log: scratchpad numbers_evidence.md).

Be honest about the record. The lead session that reached 19/20 (one seed) in 74 min read model_info.md first (22:40:59).
It then found and used the real checkpoint, the lerobot source and a MuJoCo sim (22:41–22:54). Every fact that
its two winning edits needed is also in model_info.md + model.txt, and that is checked below (§3–§5). The Codex
parent worked from the anonymised dumps and found the same pad-zero edit from the L1 numbers in 8 minutes
(Codex 04:27–04:30).

---------------------------------------------------------------------------------------------------------------

## 0. The rules (read these if nothing else)

1. **Structure before neurons.** Interface dims, padding, loss masks, normalisation and the sampling loop
   produced the biggest wins. The head/neuron line plateaued for ~14 h at 5/10 (Codex 10:18 → Nov 7). The
   structure line reached 17/20 in 47 min and 19/20 (one seed) in 74 min (b078391b 22:40 → 23:54).
2. **Read the scorer's source before you trust a single number from it.** `eval_output.py` prints a step function
   of how many times it has been called (THRESHOLDS = 1, 23, 50, …). Two identification agents called it 27
   times between them. One of them read the jump to "2" on call 23 as progress (ID-agents 04:20, 04:44).
3. **Prove the numbers you edit are the numbers you test.** Match hashes or statistics between the text dump and
   whatever the harness loads. The cached checkpoint differed from model.txt (Codex 04:24). The lead's step-6000
   stats (action std[5] 40.59) differ from model_info.md's (24.68) (b078391b 22:54).
4. **Every edit is algebra you can verify offline to float precision.** If you cannot write the identity the edit
   preserves and the one quantity it changes, it is not ready.
5. **Count rank and function, not just values.** The exact deep8 translation touches 520,083 values and is rank 1
   (np). The rejected "controller" touched 214,284,189 (Codex 05:53). A value budget alone ranks these wrongly.
6. **Get the signs from the update rule, every time.** The sampler is `x ← x − 0.1·v`, so raising the final
   action means LOWERING v. One child reversed this (parent_review.json, recovery_feature_selectivity).
7. **Probe with an effect size that can show up.** ±10 mm cube moves on 4–8 frames could not resolve anything
   (Haiku children, 17:02–23:34). The deciding probes compared whole-chunk statistics over 12+ observations × 6–8
   noise draws (b078391b p2–p5).
8. **Check the test with the demonstrations.** If the demos fail the test, the test cannot rank edits (Codex
   04:48: demos dropped the cube by 14–18 s under the 18 s rule).

---------------------------------------------------------------------------------------------------------------

## 1. Read the brief and the files (≤10 min)

1. Read the whole description. Read every script you are handed (`eval_output.py`: §7.3). `head -c` the large
   dumps. Never `cat` a multi-GB file. Learn the format first: model.txt is
   `## <name> F32 shape=[...]` followed by one row per line, dim 0, row-major (model.txt l.1–4).
2. Run `editstack decode`. On model.txt it took 0.35 s and flagged the three padded interfaces without being
   asked (303f0667 Nov 11 02:26:59):
   - `L1.weight: cols split at 6: [0:6] vs [6:32] differ in norm (2.78 vs 3.36, effect 9.5)`
   - `L2.weight: rows split at 6: (0.855 vs 0.644, effect 3.6)`
   - `L5.weight: cols split at 6: differ in kurtosis (1.99 vs 1.81, effect 5.3)`

   Treat every SCAN line as a question you must answer (§3), not as a finding.
3. Write down the I/O contract as stated, and mark each item as *stated* or *verified*. From model_info.md:
   - L5 input: 32 floats, 0–5 used, 6–31 = 0.
   - L1 input: 50×32, 0–5 noisy values, 6–31 "noise only".
   - L2 output: 50×32, "only 0–5 used".
   - Normalisation `(x−mean)/std`, with SEPARATE stats for L5 and for L1/L2.
   - Inference: 10 Euler steps `x ← x − 0.1·v`, τ = 1 → 0.1. Training target `x_τ = τε + (1−τ)y`, `v* = ε − y`.
   - Time features `p_k = 0.004·1000^(k/359)`, sines first, then cosines.

---------------------------------------------------------------------------------------------------------------

## 2. Map anonymous tensors to roles (≤20 min)

### 2.1 Shape algebra
- Group tensors by shape. Repeated blocks appear as families with identical shape tuples (12 × Stack A, 16 × B,
  16 × C).
- Every linear layer is `[out, in]`. Chain them: an `out` must equal the next `in`. In the archetype, L1
  (720×32) → L3 (720×1440 = [a(720) ; t(720)]) → SiLU → L4 (720×720) → … → L2 (32×720). The 1440 = 720 + 720 split
  of L3's input is in the brief. Without a brief, a fan-in that is exactly twice a neighbour's width is a concat.
- Small dims (32 next to 720/960/2048) are interfaces. That is where padding lives.
- Odd shapes identify mechanisms. K/V widths of 320 = 5 heads × 64 against Q 960 = 15 × 64 means grouped-query
  attention (3 queries per KV head). 320×320 "K/V" tensors in alternate layers are cross-attention
  re-projections. 12288 = 16·768 means a 4×4 patch regroup. 1024×768 is a 32×32 position table.

### 2.2 Parameter accounting (a free fingerprint and a dead-weight map)
- Sum all shapes. 450,046,176 identified SmolVLA-450M from the numbers alone (ID-agents 02:37).
- Subtract what cannot reach the output: unused heads (L151, 49280×960), the final norm of the prefix stack (L298),
  the last prefix layer's Q/O/N2/G/U/D (its K/V are consumed but its output is not), and input columns that are
  always zero (L5[:,6:]). Result: 393,494,496 live scalars, and the difference is exactly
  47,308,800 + 960 + 9,216,960 + 24,960 = 56,551,680 (ID-agents 04:45, recomputed np). Never spend time or budget
  on dead parameters. Use them as free slots only if you can prove their inputs are live.

### 2.3 Verify the mapping numerically
Do not trust a role until a number confirms it:
- **Re-implement one block in numpy and match a known output.** With no activations available, use invariants:
  - A norm-weight vector of all ones means a pre-norm gain that was never moved from 1 (expert norms were exactly
    1.0: ID-agents 02:33; Haiku 17:05 for L150, L51, L67, L115).
  - A position table should be smooth along its grid.
  - Tied tensors are bit-identical (`decode` lists them; L151 vs L153 are NOT tied).
- **Round-trip test for any edit identity** (§5): build the edited and unedited sub-network and check that the
  function you claim is preserved, is preserved to ~1e-9. If the identity fails, your role map is wrong.
- **Norm/RMS profile across depth.** It is flat in this expert (q/o/MLP RMS 0.0200–0.0206 in every layer,
  MODEL_MAP l.128ff), so weight norms do not localise function. Do not hunt layers by weight norm.

---------------------------------------------------------------------------------------------------------------

## 3. Read the statistics (≤15 min): what weights reveal without activations

Default `nn.Linear` init is `U(−1/√in, 1/√in)`. So the std is `1/√(3·in)`, the kurtosis 1.8, the expected
column norm `√(out/(3·in))` and the row norm `1/√3 = 0.577`. Compare every interface slice with that.

| slice (model.txt) | measured (np) | init expectation | reading |
|---|---|---|---|
| L5[:, 6:32] (state pad) | absmax **0.1768**, std 0.1021, kurt 1.80, col norm 3.08–3.25 | bound 1/√32 = 0.1768, std 0.1021, col norm 3.16 | **untouched init.** Input always 0 → zero gradient. Dead weights, harmless |
| L5[:, 0:6] (state) | absmax 0.293, kurt 2.03, col norm 2.96–3.56 | – | trained |
| L1[:, 6:32] (action pad in) | col norm **3.32–3.40**, absmax 0.253 | 2.74, bound 0.177 | **trained harder than the real columns.** The pad inputs carry τ·ε in training, so the net uses them (they reveal the noise level). Live input, not inert |
| L1[:, 0:6] (action in) | col norm 2.62–2.94 | 2.74 | trained |
| L2[6:32, :] (action pad out) | row norm 0.636–0.652 (tight) | 0.577 | homogeneous, unlike real rows. **Outputs never in the loss** |
| L2[0:6, :] (action out) | row norm 0.70–1.21 (spread) | 0.577 | trained; row 5 (gripper) stands out (also ID-agents 02:36) |
| L3 time half, columns by period | 0.17 at the shortest periods → 0.43 at the longest | 0.41 | high-frequency τ features are nearly unused |
| expert RMSNorm weights | exactly 1.0 | 1.0 | gain never trained. A scalar gain there is an exact uniform scale of that norm's output |

Rules of thumb:
- **Init-like slice with always-zero input:** dead, ignore it.
- **Init-like or homogeneous slice on an output that nothing supervises:** its value is junk. Find out where junk
  goes next (fed back? discarded?).
- **Slice that grew MORE than the real channels:** the network depends on it. Removing it changes behaviour, so
  look for the reason.
- **Kurtosis alone is weak.** All of L1 sits at kurtosis 1.83, real and pad alike (np). Use norm splits, absmax
  vs the init bound, and spread across rows together.

Measure the functional weight of a slice with a numpy forward pass through the sub-block you can rebuild. With
unit-variance inputs at τ = 0.1, resampling the 26 pad channels moves the action-token embedding (L4 output) by
0.742 mean |Δ|. Resampling the 6 real channels moves it by only 0.455 (np). That single number says the padding
dominates the action tokens.

---------------------------------------------------------------------------------------------------------------

## 4. From the description to defects: the inference, written out

### 4.1 Jitter = untrained padding fed back through the sampler
Chain every step to a sentence in the brief or a number:
1. "L2 output 50×32, only 0–5 used" means the loss covers v[:, :6] only, so v[:, 6:] is never trained (stats §3:
   L2 pad rows homogeneous).
2. "L1 input 6–31 noise only" plus the sampler `x ← x − 0.1·v` over all 32 channels means the untrained v_pad
   is fed back into x_pad at every one of the 10 steps.
3. The training target `x_τ = τε + (1−τ)y` with y_pad = 0 means that in training, x_pad = τ·ε. Its std is τ,
   which is 0.1 at the last step.

   At inference, x_pad drifts under junk v_pad. Measured with the full model: std 0.59 at τ = 0.1 vs 0.1 in
   training, about 6× out of distribution (b078391b Nov 6 00:30; MODEL_MAP l.428).
4. L1 pad columns are full strength (§3: 3.36 vs 2.79), so that out-of-distribution noise enters every one of
   the 50 action tokens.
5. ε is drawn independently per timestep, so the kick is independent per row of the chunk. That is frame-to-frame
   wiggle *inside* a chunk, which is the reported "jitter". The lead confirmed it as a within-chunk second
   difference (p1, b078391b 22:55).
6. Fix: `L1[:, 6:] = 0`. The network then becomes exactly a 6-channel head, and v_pad can no longer reach anything
   that matters.

Corrections the lead had to make, so you don't repeat them:
- The pad *inputs* are seen in training. Only the pad *outputs* are unsupervised.
- Zeroing the initial pad noise is NOT equivalent to zeroing the weights (outputs differ by up to 0.22): after
  step 1 the junk outputs refill the channels (b078391b Nov 6 00:30).
- Variance attribution under-predicted the effect. Pad noise looked like the smaller share (d2 0.16–0.68 vs
  0.38–1.85 for joint noise, p3 23:13). Codex likewise judged that "most remaining variation comes from the six
  real joint-noise inputs" (Codex 09:34). Yet the ablation cut d2 about 5× (0.085–0.298 vs 0.465–1.976, p5 23:19).
  **Ablate. Do not apportion.**

### 4.2 The flow-matching facts that become edit formulas
From `x_τ = τε + (1−τ)y` and `v* = ε − y`:
- `v* = (x_τ − y)/τ`, so the ideal Jacobian is `∂v/∂x = I/τ`. Measure `τ·∂v/∂x` along random directions on
  the real channels; it should be ≈ 1. Measured: 1.00 at τ = 1, falling to 0.785 at τ = 0.1 (p4, b078391b 23:13).
  A value below 1 means the last steps under-contract noise.
- The sampler gives `x_0 = x_1 − 0.1·Σ_{steps} v`. A constant change δv on every step moves the result by
  exactly `−δv` (10 × 0.1 = 1) *if the network does not react*. It does react, because the inputs moved. Hence
  the input-side compensation in §5.2.
- Feature k = 0 (period 0.004) is ≈ constant on the schedule τ ∈ {1.0, …, 0.1}: sin ≈ 0, cos = 1
  (ID-agents 02:44; np). Never use it to carry τ.
- Normalisation: an edit of Δ raw units is `d = Δ/std_action` on the L1/L2 side and `d_s = Δ/std_state` on the
  L5 side. The two std tables differ (e.g. index 5: 24.68 vs 22.78). Using one for both is a silent error.

---------------------------------------------------------------------------------------------------------------

## 5. Exact edit derivations that worked

Notation: L1 = (A, a), L3 = [W_a | W_t] with bias b3, L2 = (B, b2), L5 = (S, s), φ(τ) = 720 time features.

### 5.1 Zero the pad columns (`pad0`)
`A[:, 6:32] = 0`. That is 18,720 values in 1 tensor.
- Check: the L4 output is independent of x[:, 6:] to 0.0 (np).
- Result: within-chunk d2 0.465–1.976 → 0.085–0.298 offline (p5). Sim jitter 0.910 → 0.192, held ≥1 s 8/20 → 17/20
  (b078391b 23:27).
- Codex found the same edit at 04:30 (~44% lower action-step RMS, Codex 05:15) but scored 0/5 under a test the demonstrations also failed
  (§0 rule 8).

### 5.2 Exact translation of the action space by Δ (`translate_actions`, "deep8")
Goal: the sampler outputs y + d instead of y, with nothing else changed.

Under the new policy the noisy sample is `x' = τε + (1−τ)(y+d) = x + (1−τ)d`. We need `v'(x', τ) = v(x, τ) − d`.
1. Input. `A(x' − (1−τ)d) + a = A x' + (a − A d) + τ·A d`.
   - Constant part: `a ← a − A d` (720 values).
   - The τ part, `τ·A d`, must enter L3 as `τ·W_a A d`. Supply it through the time features: find c with
     `φ(τ)·c ≈ τ` and set `W_t ← W_t + (W_a A d) ⊗ c` (rank 1, 720×720 = 518,400 values).
2. Output. `b2[:6] ← b2[:6] − d` (3 nonzero entries for Δ = [0, 1.52, 1.52, −3.11, 0, 0]).
3. State (optional, closed loop). Make the network perceive the shifted state in its usual frame:
   `s ← s − S d_s`, with `d_s = Δ/std_state` (960 values).
4. Induction check. If `x'_τ = x_τ + (1−τ)d`, then
   `x'_{τ−0.1} = x'_τ − 0.1(v − d) = x_{τ−0.1} + (1−τ+0.1)d`. At τ = 0 the shift is exactly d.

Fit c (the lead's recipe, reproduced exactly):
```python
k = np.arange(360); w = 2*np.pi / (0.004 * 1000**(k/359))
phi = lambda t: np.concatenate([np.sin(np.outer(t, w)), np.cos(np.outer(t, w))], 1)
taus = (np.float32(1.0) - np.float32(0.1)*np.arange(10, dtype=np.float32)).astype(float)  # float32 schedule
grid = np.concatenate([np.linspace(0, 1, 2001), np.tile(taus, 200)])   # weight the 10 inference taus
G = phi(grid); c = np.linalg.solve(G.T @ G + 1e-6*np.eye(720), G.T @ grid)
# |c| = 0.1144, max error at the 10 taus 1.72e-8, on all of [0,1] 2e-6
```
- Do not use the minimum-norm exact interpolant `pinv(phi(taus)) @ taus`. It is exact on the schedule, but off
  it the error is 0.51, so any schedule mismatch breaks it (np).
- Do not use a sparse c to save values. 10–40 time columns can be exact, but |c| = 832–109,642, which is
  ill-conditioned and fragile in float32/bf16 (np).

Verification without any model code (np, model.txt):
- `max |emb_edit(x + (1−τ)d, τ) − emb(x, τ)|` over the 10 τ's is 1.7e-9 in float64 and 1.2e-8 with float32
  storage.
- Euler bookkeeping with any velocity that depends on x only through that embedding returns a final shift of
  exactly [0, 1.52, 1.52, −3.11].

With the full model (b078391b 23:35):
- With the state term off, the open-loop chunk mean shifted by exactly 1.5/1.5/−3.1 and d2 was unchanged.
- With the state term on, the open-loop shift is smaller (0.40/0.83/−1.13). This is expected: the network now
  believes the arm is elsewhere. Only closed loop tells you the net effect.
- Sim: 17/20 → 19/20 (23:54).

Choosing Δ is not numbers-only. Δ came from arm geometry: an 8 mm deeper close, solved through the MuJoCo
Jacobian, median [−0.03, 1.52, 1.52, −3.11] (jac2, 23:29). Without a sim or kinematics, take Δ from the user or
from a measured offset in executed actions. Never guess it from weights.

### 5.3 Per-axis gain about a centre C (`scale_axis`)
Goal: commands `y' = C + k(y − C)` on one axis, while the state is perceived as `C + (s − C)/k`. Holding still
is preserved. Every image-driven move on that axis is multiplied by k.
- In normalised units `ŷ' = kŷ + β`, with `β = (1−k)(C − μ)/σ`.
- Input column `A[:, j] /= k`. If β ≠ 0, apply the §5.2 translation terms with `d = β·e_j`.
- Output row `B[j] *= k`, and `b2[j] = k·b2[j] − β`.
- State column `S[:, j] /= k`, plus a bias term for `C − μ_state`.
- Exact except one thing: the network now sees that axis's τ = 1 noise with std 1/k. That effect is ≤ 0.06 units
  (MODEL_MAP l.466).
- With C = action mean, β = 0 and the time MLP is untouched.
- Verify before any sim. Run the base model on the analytically transformed input (state mapped, noise ÷ k),
  transform its output, and compare with the edited model (`check_scale.py`). Max diff 0.03–0.06 raw = float
  rounding (b078391b Nov 6 23:38).
- Cost: an affine gain acts everywhere. pan×1.7 fixed lateral cubes (1/12 → 29/36) but cost in-distribution
  picks (73/80 → 63/80, MODEL_MAP l.466). Plan for gating if the fix should act only out of band.

### 5.4 Time shift (exact, but it did not help)
Rotate each (sin_k, cos_k) column pair of W_t by `w_k·δ`. The network then sees τ − δ exactly
(`sin(w(t−δ)) = sin·cos(wδ) − cos·sin(wδ)`). It was motivated by the 0.785 contraction. On top of pad0 it
overshot, so it was dropped (p5, 23:19–23:24). Keep it as a tool, not a fix.

### 5.5 Why the naive edits failed (numbers)
- **Global output gain ×1.2** (`B[:6], b2[:6] *= 1.2`).
  - Offline: noise residual coefficient 0.027 → 0.005 and d2 −58% (p2 22:59–23:06), but the chunk *mean* also
    moved by 0.5–2.2 raw units per joint.
  - Sim: jitter 0.910 → 0.396, but held 8/20 → **6/20**, and tip–cube offsets at close grew to 18–32 mm
    (g120, b078391b 23:10).
  - A gain on v scales the signal along with the noise. The offline mean shift predicted the failure; the sim
    run was avoidable.
- **Bias nudge for a deeper grasp** (`b2 −= Δ/(0.1·σ)`, which assumes only the last step counts).
  - Target 1.5/1.5/−3.1. Got 6.2/8.5/−14.9, i.e. 4–6× (b078391b 23:29).
  - The assumption was off by 10× (all 10 steps apply the bias), and the network partly fought the
    off-distribution inputs.
  - The exact §5.2 version hit the target to the digit 6 minutes later. Codex hit the same wall ("didn't account
    fully for denoising time", Codex 06:52; "deeper-grasp shift hurt", 09:24).

---------------------------------------------------------------------------------------------------------------

## 6. Edit-size discipline

- **Report per edit:** tensors touched, values changed, rank of each delta, relative Frobenius change, and the
  one functional claim with its offline check. Examples:
  - pad0: 1 tensor, 18,720 values, relative change 0.93 of L1 (np), "6-channel head" exact.
  - deep8: 4 tensors, 520,083 values, rank 1 in the time half, relative changes 0.161 / 0.0106 / 0.144, "shift
    by Δ" exact to 1e-8.
- **Values-changed budgets are a proxy.** Negotiate them with this table:
  - A 214,284,189-value rewrite (47.6% of the model; 214,258,308 values zeroed; vision disconnected) was rejected by
    the user as a hard-coded controller rather than an edit (Codex 05:53–06:18).
  - Codex then worked under <100,000 values: 2,260 → 1/10, 2,979 → 2/10, 3,699 → 3/10, 88,332 → 5/10
    (Codex 07:28–10:06).
  - The exact translation would not fit under that cap, though it is rank 1 and changes the time half by 1%.
    When a budget forbids an exact low-rank edit, say so and propose counting rank instead. Do not swap in an
    inexact sparse substitute.
- **The input-sensitivity audit catches controllers.** Change an input that should matter (cube image) and require
  the output to change. The controller produced all 540 commands identical across five different cube placements
  (Codex 06:28).
- **Audit every changed value.** Diff edited vs original over all tensors and list what changed beyond the
  intended slices. Codex kept `exact-changes.json` (old/new for every scalar), sha256 of base and fixture, and a
  reload check of all 500 tensors (experiment_index.json, progress.json). The lead kept one `ed/<name>/EDIT.py`
  per variant, applied in memory to the untouched base, and exported with SHA256SUMS. Use `editstack apply` /
  `export` for both.
- **Provenance:** every variant is (base hash, edit code, parameters, fit data path). Never hand-edit numbers in a
  dump file. Regenerate from code.
- **Free choices:** prefer edits on dead or interface parameters (pad columns, unit norm gains, biases) over edits
  inside trained blocks. Gated circuits in least-used units come later (EDIT_COOKBOOK §C).

---------------------------------------------------------------------------------------------------------------

## 7. What numbers alone can verify, and what needs code, a forward pass, or a closed loop

### 7.1 Verifiable from the dumps plus numpy, with no code
- Role map: shape algebra, the parameter count, dead-parameter accounting.
- Init and pad signatures (§3). Unit norm gains. Tied tensors.
- **Edit identities** on the sub-block they touch: pad0, translation, scale, time shift, function-preserving
  rescale pairs. Rebuild only L1 → L3 → SiLU → L4 (all in model.txt). Everything downstream sees the action
  tokens only through that output, so equality there proves equality of the whole network (§5.2).
- The τ-fit error. Float32/bf16 storage error of the edited tensors (re-cast, re-run the identity: 1.2e-8 here).
- Functional weight of interface slices via sub-block sensitivity (§3: 0.742 vs 0.455).

### 7.2 Needs a full forward pass (code or a full numpy re-implementation)
- Effect sizes on outputs: d2 / within-chunk roughness, the chunk-mean shift, the τ·∂v/∂x contraction.
- These decided pad0 vs time_shift and predicted the g120 failure.
- Without code, re-implement the stacks from model_info.md in numpy (~450 M params, fp32, about 2 GB). That costs
  about an hour and is worth it if no code exists. Validate the re-implementation with an invariant first.
  Example: an edit that must be a no-op (function-preserving rescale) gives identical outputs.

### 7.3 Needs a closed-loop test (sim or real)
- Whether the score improves. pad0 cut jitter in both lines, but success moved only where the test could see it:
  8 → 17/20 for the lead, 0/5 for Codex.
- How large Δ should be (kinematics).
- Whether a state-side term helps (closed-loop only, §5.2).
- Real hardware only after the sim is validated and you can state the expected outcome.

### 7.4 Do not be fooled by a scorer
- Read its source first. `eval_output.py` ignores `--result-explanation` and prints how many thresholds the call
  count has passed. The ledger lives outside the folder in base85+zlib+reversed-base64 (eval_output.py l.20–60).
- If no source is available:
  - Submit the same text twice. Submit nonsense. Submit text with obvious errors.
  - Treat a score that depends on call count or wall time and not on content as noise.
  - Never let several agents share one scorer whose state persists. The two ID agents moved each other's scores.
- Never iterate content against a scorer that gives no criterion-level feedback. Stop and report "blocked: scorer
  is uninformative" (ID-agents 02:59).
- Never optimise wording to a score. A behaviour claim needs a behaviour test.
- Hold your own harness to the same standard. Baseline runs must be deterministic (paired seeds; reruns identical,
  b078391b 22:52), and the demonstrations must pass the test.

---------------------------------------------------------------------------------------------------------------

## 8. Fast path (time-boxed recipe)

Lead timeline for reference: harness and baseline 7 min, first probe 14 min, pad0 in sim 47 min, deep8 74 min
(b078391b 22:40 → 23:54, including ~14 min lost to a disk-full crash).

| # | box | do | stop / output |
|---|---|---|---|
| 1 | 0–10 min | Read brief + every script. `editstack decode`. Check the format. Confirm the dump hashes/stats match what the harness loads | ARCHITECTURE.md stub; list of SCAN questions |
| 2 | 10–25 | Shape algebra, parameter count, dead-parameter list. Write the I/O contract (stated vs verified) | role table L-number → role, each with a check |
| 3 | 25–40 | §3 statistics on every interface slice. Rebuild the smallest fused block (here L1 → L3 → L4) in numpy | numbers for each pad/init/norm hint |
| 4 | 40–55 | Chain brief sentences + statistics into defect hypotheses (§4). For each: which tensor, what identity, what predicted effect | ≤3 hypotheses with predicted numbers |
| 5 | 55–70 | Derive the exact edit (§5). Verify the identity offline (≤1e-6). Measure the effect on a forward pass if you have one | edit code + verification output |
| 6 | 70–90 | Closed loop: baseline vs edit on paired seeds; then geometry-driven Δ (§5.2) if the failure is positional | score + per-episode geometric errors |
| 7 | after | Stack the next exact edit on the best one. Only then readout fits / gated circuits (EDIT_COOKBOOK) | — |

Box rules:
- Skip a closed-loop run whose offline numbers already predict failure. g120's offline mean shift did
  (b078391b 23:06).
- Edit in memory from code. Do not write checkpoint copies (each one costs 865 MB; this filled the disk at 23:40).
  Do float64 algebra on CPU copies, then copy back in the storage dtype (MPS has no float64: the error at 23:41).
- One owner per hypothesis. The lead does steps 1–5 alone. Delegate only from step 6 on, and only with a
  numeric target (AGENT_FLEET.md).

## 9. Slow path: what made the Codex/Haiku line slow (avoid)

| mistake | evidence | instead |
|---|---|---|
| Discarded the right first edit because the test could not see it | pad-zero at 04:30 → 0/5. Demos fail the 18 s rule (04:48). Padding variants later scored only 3–4/10 at 92–107 k values | validate the test with demonstrations; score staged metrics (jitter, reach, grasp) |
| Head/neuron/RoPE/Q-K hunting before readout gains | 133 full 10-case suites, 1,402 rollouts. 5/10 plateau from 10:18 Nov 5 to the end. Distribution 0:17, 1:30, 2:38, 3:29, 4:17, 5:2 (experiment_index.json) | structure first (§4); then measure gains/offsets at the readout |
| Forbade closed-form fitting as "not manual" | child briefs: "no regression/fitting observations or target actions into weights" (299037cf) | exact algebra and least-squares readouts are manual edits. Fitting c for φ(τ)·c ≈ τ fits a known function, not data |
| Diagnose-only children with no authority to edit or score | 27 Haiku sessions, 3–15 min each. 18 reviewed, 0 accepted (parent_review.json) | give an owner a numeric target and the right to produce a variant |
| Re-deriving context in every child | all 27 re-mention or re-verify the 88,332 recipe and reproduction gates | write facts once in MODEL_MAP/ARCHITECTURE; children read, not rebuild |
| Probes too small to resolve anything | ±10 mm cube moves, 4–8 case-frames (planar/depth children) | probe effects ≥ the error you need to fix; ≥12 observations × ≥6 noise draws |
| Sign and axis errors | flow-to-action sign reversed (recovery_feature). The "depth" probe was a near-vertical axis (depth_response) | derive signs from `x ← x − 0.1·v`; state axes in a frame you checked with one known move |
| Garbled briefs | whitespace stripped: "unchanged-fixture10/10", "under100000" (299037cf) | send briefs as files; re-read them |
| Attributing variance instead of ablating | "most remaining variation comes from the six real joint-noise inputs" (Codex 09:34) vs the 5× ablation result (p5) | ablate the slice, measure, decide |
| Hand-wired controller to hit the number | 214 M values, vision zeroed, identical outputs across placements (05:53–06:28) | input-sensitivity audit before reporting; report fixed-test overfitting as a failure |

## 10. Minimal numpy kit (copy, then adapt)

```python
import numpy as np, re
def load_dump(path):                       # '## name dtype shape=[...]' + one row per line
    T, name, shape, rows = {}, None, None, []
    for line in open(path):
        if line.startswith('## '):
            if name: T[name] = np.array(rows, float).reshape(shape)
            name = line.split()[1]; rows = []
            shape = [int(s) for s in re.search(r'shape=\[([^\]]*)\]', line).group(1).split(',')]
        elif line.strip() and not line.startswith('#'):
            rows.extend(map(float, line.split()))
    if name: T[name] = np.array(rows, float).reshape(shape)
    return T
kurt = lambda x: ((x - x.mean())**4).mean() / x.var()**2
def init_report(W):                        # W [out, in]; compare with U(-1/sqrt(in), 1/sqrt(in))
    out, inn = W.shape; bound = inn**-0.5
    return dict(bound=bound, absmax=abs(W).max(), col_norm_init=(out/(3*inn))**0.5,
                col_norms=np.linalg.norm(W, axis=0), row_norms=np.linalg.norm(W, axis=1), kurt=kurt(W))
# For an identity check: rebuild the sub-block, apply the edit, compare to 1e-6, and also after casting
# the edited tensors to the storage dtype (float32 / bf16).
```
For model_2.txt (5.9 GB): find block offsets with one streaming pass (`grep -b '^## '`), save them, then
`seek` to the tensors you need. Never load the whole file into memory (or the context).
