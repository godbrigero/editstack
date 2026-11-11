# Approach playbook: from symptom to edit

Use this after ANALYSIS_GUIDE.md has given you a measured symptom. Find the row, run the confirming probe, and
apply the edit family it names. Always use the highest rung of the exactness ladder that works:
**exact algebra > closed-form fit > gated circuit** (recipes: EDIT_COOKBOOK.md).

## Symptom -> cause -> confirm -> edit

| symptom (measured) | likely cause | confirming probe | edit family | past case |
|---|---|---|---|---|
| jitter / wiggle inside each output chunk | inputs fed at inference but never supervised (padding, unused slots) inject noise | zero the suspect input columns in a single-chunk probe; jitter drops | **exact**: `ops.zero_cols` | pad0: jitter 0.91 -> 0.19, 8/20 -> 17/20 |
| consistent offset (always short / long / high by d) | systematic bias in the output space | signed error vs task variable is flat with offset d | **exact**: translate the output space by the Jacobian step (`compensate_input_shift` + biases; flow heads need the tau term) | deep8: closes 8 mm deeper, 19/20 |
| error grows with distance from the trained range (slope, not offset) | read-out gain too low out of distribution; commands compress toward the hold | `fit.gain` < 1 on that axis; decodability R^2 still high | **exact**: per-axis scale with perception compensated (`scale_axis`), then **gate** it to out-of-band | pan x1.7: near 5/12 -> 31/36; gated p17gs kept ID at 38/40 |
| a fix helps out-of-band but costs in-band | the fix is active where the model already worked | per-region gain before/after; in-band over-aim (was 1.5x) | **gated circuit**: multiply by an out-of-band feature or by (planned - state) so it is 0 in-band | pancost diagnosis -> approach's gated version |
| right place, wrong orientation (grip fails after reach) | an under-gained secondary axis (roll) | failure taxonomy + roll error at close; gain ~0.03 | **gated circuit** on that axis (exact product pair) | roll servo: +3 near picks (p17gshr) |
| image-based correction too weak | the image term of the correction is small relative to the state term | ridge split of the correction into image and state parts | **closed-form**: rank-1 fit amplifying the image part (and damping the state part) | sp6: near 33/36 reach |
| everything broke after a sensor / camera / preprocessing change | inputs moved off the trained distribution | paired renders old vs new; per-layer feature shift | **closed-form**: `fit.paired_map` + `ops.fold_input_map` at the layers where the map is most linear | cf2: ID reach 6/40 -> 59/60 |
| the model depends on a sim-only artifact | spurious feature (shadows, reflections) | render with and without the artifact | gated / closed-form removal, but check it doesn't fight earlier edits | shadow: helped sp6 but hurt p17gshr (double correction) |
| no attempt / stalls far from the target | the object is outside the camera view or the policy never commits | log what each camera sees; time to first approach | circuit that keeps moving until the feature appears ("seek") | sk5p: no-turn test 10/24 -> 22/24 |

## Choosing between candidate approaches

1. **Can it be exact?** If the fix is "change units/offset/gain on an axis", or "remove an input", or "map inputs
   back to the trained distribution", it can be done algebraically. Do that.
2. **Is the needed signal linearly available?** Run `fit.cv_r2` at the layer you would edit. Below ~0.8, a
   linear fit or circuit there will not work. Move earlier or later.
3. **Must it be silent in-distribution?** Then it needs a gate. Find a feature that is exactly 0 (or separable)
   in-band, and check the gate's false-positive rate on the id suite's states.
4. **Rank.** Start at rank 1. Raise it only if the held-out fit R^2 improves.

## Approaches that did NOT pay off (don't start here)

- Changing execution instead of the policy: re-plan every 10 or 25 steps, temporal ensembling (worse: 9/24 and
  15/24 vs 21/24).
- Editing during training (training cancels the edits: 39/80, the same as control at 37/80; editing the finished
  checkpoint once gave 60/80).
- Hard-coding a controller by changing hundreds of millions of values (rejected: not an edit of the model's
  behaviour, a replacement of it).
- A global output gain without per-axis analysis.

## Composing approaches

Stack edits as one chain (PARENT). Before adding a link, ask: **does an earlier link already correct this
quantity?** Double correction was the main source of interference. Re-qualify after every link.
