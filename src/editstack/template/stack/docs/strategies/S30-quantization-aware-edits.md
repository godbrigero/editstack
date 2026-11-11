# S30 — Quantization-aware edits (survive the storage dtype)

Status: untested idea
Last updated: 2026-11-10

## When to use
- The checkpoint stores tensors in bf16/fp16/int8 (SmolVLA's expert MLP is bf16) and your edit is a SMALL delta on a
  large weight, or a circuit with very small up/down weights.
- Offline probes ran in float32/64 but the exported model behaves differently.

## Preconditions & confirming probes
- `model/INVENTORY.md` dtypes. bf16 has 8 bits of mantissa: relative step 2⁻⁸ ≈ 0.4 %. A delta smaller than
  ~0.2 % of the element's magnitude rounds away.
- Measure: fraction of edited elements whose stored value is unchanged after casting (`editstack apply` delta stats).

## Procedure
1. Compute the edit in float64 (editstack does), then cast a copy to the storage dtype yourself
   (`np.asarray(W[name]).astype(ml_dtypes.bfloat16)` or torch) and re-run the offline probe on the CAST weights.
2. If the effect shrinks: (a) move the change to a place where it is not small relative to the weight (a repurposed unit,
   S05, whose rows you overwrite entirely); (b) rescale a unit pair: gate × a, up × b, down / (ab) keeps the product but
   moves magnitudes into a better-represented range; (c) for biases kept in fp32, prefer bias edits.
3. Record the realised/nominal ratio after casting in the edit docstring.

## Verification
`editstack export` verifies bit-exactness of the export vs the applied variant; the offline probe must use the exported
dtype. Then rungs b–d.

## Expected effect & cost
Unknown as a strategy. Related facts: circuits noted ~0.4 % bf16 rounding; realised gains were 0.5–0.8× nominal (partly
the norm approximation); in the hybrid, ~1e-3 bf16 up/down weights were the ones that drifted under Adam.

## Failure modes / interactions with other strategies
- Sharp gates need large gate rows (hybrid tipgate rel. change 48.5); check overflow/saturation in fp16.

## Evidence log
- 2026-11-07, circuit bf16 caveat. Evidence: OLD/agents/circuit/REPORT.md §1; HYB/EDIT_LEDGER.md "c4_tipgate_refresh" diagnosis.
