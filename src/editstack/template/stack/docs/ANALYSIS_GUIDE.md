# Analysis guide: how to understand a model well enough to edit it

Goal of analysis: turn "the model fails at X" into **"quantity Q is wrong by N at place P, because of mechanism M"**.
Only that sentence licenses an edit. Every probe below answers one piece of it, cheaply, with a number.

The order matters. In past runs, the route through structure, then code, then cheap probes reached 17/20 in 47 min (19/20 at 74 min on one seed).
A head/neuron interpretability hunt that started deep inside the network sat at 5/10 for ~14 h. (The two used
different checkpoints and tests, so treat this as strong judgement, not a controlled result: LESSONS.md.)

---------------------------------------------------------------------------------------------------------------

## Step 1. Recover the forward pass (before looking at any activation)

Read in this order. Write each fact into `model/ARCHITECTURE.md` with `[verified: file:line]`.
1. **`model/INVENTORY.md` + `model/SCAN.md`** (from `editstack decode`): sizes, repeated blocks, dtypes, and
   suspicious slices. Treat each SCAN hint as a question to answer.
2. **The model code**: forward(), the input embedding of every modality, attention masks, the output head.
3. **The training code**: the loss and **its masks**, data normalisation, augmentation, which inputs varied
   during training and over what range (the *in-distribution envelope*).
4. **The inference code**: sampling loop (steps, step size, what is fed back), chunking, post-processing, clipping.
5. **The deployment code**: how many predicted steps are executed, at what rate, with which sensors.

If names are anonymous (L1, L2, ...), recover the wiring from the shapes plus any description you have. The
astra-style `model_info.md` table mapping L-numbers to stack/layer/role is the target format. Group tensors by
shape (INVENTORY.md does this) and match them to the description. Then confirm with an equivalence test:
re-implement one block in numpy and check that it reproduces the model's activations on one input.

**Things that are easy to miss and were decisive before:**
- Outputs padded to a fixed width where the loss only covers the first k channels (`losses[:, :, :6]`). The
  padding then carries noise that is fed back at every denoising step.
- The difference between training-time and inference-time inputs (noise level at the last step was 6x the
  training noise on the padded channels).
- Which camera stream / resolution / pose the real system uses compared with the sim.

## Step 2. Measure the failure (from eval artifacts, not from watching videos)

- Use **staged metrics and outcome codes** (stack.json). Count failures per code over at least one full suite:
  "49 of 69 reached-but-not-picked had wrist roll > 25 deg off" pointed straight at one fix.
- Use **signed geometric errors per episode** (radial / sideways / height at the decisive moment; `geo.py` showed
  the gripper closing 6-10 mm short). Plot the error against the task variable (e.g. cube x). A slope tells you
  "gain too low". An offset tells you "bias".
- **Jitter / smoothness**: second differences of the commanded actions, within a chunk and across chunk boundaries.
- Before trusting a trend, check its size against the seed-to-seed spread.

## Step 3. Locate it: the probe catalogue (cheapest first)

All probes run offline on **recorded observations** (saved by `run_episode` into `runs/.../ep<i>/`). One model
process each (`st.policy(v)` takes a RAM slot), seconds per question. Write each result to MODEL_MAP.md.

| probe | question | how | past finding |
|---|---|---|---|
| **single-chunk response** | does the output move the right way when the input changes? | feed one obs, perturb one input (move the object in the image, shift the state), record the output chunk | jitter came from the padded channels: zeroing them in a probe removed it |
| **linear decodability** | is the task variable present at layer L? | capture activations, `fit.cv_r2(X, y, groups=episode)` per layer | cube position R^2 0.87-0.99 at every layer, so perception was fine and the read-out was at fault |
| **gain** | how much does the output correct per unit of error? | `fit.gain(error, correction)` across hover states | pan 0.34-0.6, radial ~0.13, roll ~0.03 per chunk, where ~1 was needed |
| **source attribution** | which input/camera drives this output? | swap or blank one camera (or one token group) and measure the output change | the first move was aimed by the front camera alone, reading forward distance, not radius |
| **path ablation** | which layers carry the signal to the output? | zero or mean-ablate a sublayer's contribution, measure the output | images reach actions only through the expert's odd (cross-attention) layers |
| **gate features** | can a later layer tell "in-band" from "out-of-band"? | ridge read-out of the condition from the MLP input of the last layers | the layer-15 MLP input linearly encoded every gate the circuits needed |
| **equivalence test** | is my understanding exact? | apply the function-preserving half of an edit and compare outputs | `scale_axis` verified offline (max diff 0.05 raw units) before any sim |
| **sensitivity to deployment differences** | does the model depend on a sim artifact? | render with and without it (shadows, camera pose) | removing arm shadows cost 6-7 picks per set |

Design rules for probes (each one was broken by a failed probe before):
- **Enough samples, grouped.** 4-8 frames at +-10 mm cannot resolve a weak effect. Use full suites' worth of
  states and group the CV by episode/placement, never by frame.
- **Signs and axes.** State the coordinate frame, and the sign convention of every quantity, in the script
  header. A reversed flow sign and a probe along the wrong axis both produced confident wrong conclusions.
- **Compare against a baseline probe** (the same probe on the original model) in the same run.
- **One question per probe, one number out.** Save the number with the script path as evidence.

## Step 4. Write the causal sentence

> "Beyond |x| > 7 cm the pan command compresses toward the hold. The per-chunk pan gain is 0.34 (needs ~1). The
> front camera alone aims the first move. Evidence: agents/x/gain.json, probes/src_attr.py output."

Then predict the effect of the fix with a number ("REACH near 5/12 -> >= 9/12") **before** running it. A
prediction that misses is information: record it in MODEL_MAP.md under negative results.

## Analysis anti-patterns (each cost hours before)

- Starting at attention heads, neurons or RoPE before checking interfaces, loss masks and read-out gains.
- Naive output-bias nudges instead of exact algebra: a naive bias shift moved joints 6.2/8.5/-14.9 against a
  1.5/1.5/-3.1 target, while the exact translation hit it.
- Tuning a global output gain (x1.2 made success worse, 8/20 -> 6/20) without knowing which axis is under-gained.
- Many small read-only agents each re-deriving the same context, forbidden to edit or score (27 agents, 0 accepted:
  the brief design guaranteed it; see AGENT_FLEET.md).
- Explaining from videos instead of measured errors.
