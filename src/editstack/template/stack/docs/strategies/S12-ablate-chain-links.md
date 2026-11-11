# S12 — Ablate chain links before promotion (marginal value of each edit)

Status: partial
Last updated: 2026-11-10

Evidence that it matters is strong (hybrid: the last four circuits added nothing). The systematic leave-one-out
procedure itself was never run in either project; only forward stage-by-stage scoring was.

## When to use
- Before promoting a chain with more than ~2 links, and whenever a link's own gain was within noise when added.
- When a chain was built on a different base than the one it is now applied to (refits, new cameras).

## Preconditions & confirming probes
- The chain is expressed as PARENT links so each prefix and each "minus one link" variant can be built in memory.
- A paired eval (same placements and seeds) and a screen suite cheap enough to run n+1 variants.

## Procedure
0. In editstack: `editstack ablate <variant> --suite <target> --seeds 0,1,2` builds and evaluates every
   chain-minus-one-link variant (`<v>~minus~<link>`) paired against the full chain; `editstack promote` reports links
   whose removal never hurt.
1. **Forward stages** (cheap, always): score prefixes m1, m2, … mn on the qualify suites (paired, ≥2 seeds).
2. **Leave-one-out** for links whose forward marginal is ≤ 1 SE: build chain-minus-link-i (edit.py with that call
   removed, everything else identical, readouts unchanged) and score paired against the full chain.
3. **Offline firing check**: for each circuit link, the fraction of eval states where its gate is on and the size of its
   output there; a link that never fires on the suites cannot be credited with a gain on them.
4. Drop links with ≤ 0 marginal unless they serve a different suite (record which: e.g. seek fires only on the
   reproduction set; shadow-off sets).
5. Record each link's marginal in its strategy file's evidence log.

## Verification
- Paired sign test of chain vs chain-minus-link; ≥3 seeds for links you keep on a small marginal.

## Expected effect & cost
- Hybrid ENDEDIT stages: m1 pad-zero 44/80 → m2 + servo 60/80 → m7 (+ 4 circuits) 60/80. The 4 circuits added 0 on the
  main suite (they helped only κ/generalisation families) and cost loss.
- Old project, measured per link: +h ID REACH 38 → 40/40; +r near PICK 29 → 32/36 (within noise); grasp roll across 4
  bases net near +3 / ID −2. +seek neutral on standard sets, +12 REACH on its reproduction set.
- Cost: n extra screen/qualify runs; cheaper than carrying dead links into every later refit.

## Failure modes / interactions with other strategies
- Links interact: two links correcting the same quantity (S14 on p17gshr) can each look good alone and bad together;
  ablation reveals it only if run on the full chain.
- With 12–20-episode sets, ±1–2 per seed is noise: don't remove a mechanism-verified link on one seed.

## Evidence log
- 2026-11-09, hybrid control/6000 ENDEDIT stages m1/m2/m7: 44/60/60 of 80. Evidence: HYB/EDIT_LEDGER.md "ENDEDIT"; HYB/agents/endedit/chain.log.
- 2026-11-07, p17gs → p17gsh → p17gshr link deltas. Evidence: OLD/agents/approach/REPORT.md results table.
- 2026-11-07, grasp roll on 4 bases. Evidence: OLD/agents/grasp/REPORT.md §4.
