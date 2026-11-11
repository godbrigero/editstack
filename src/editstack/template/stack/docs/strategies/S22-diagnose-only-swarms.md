# S22 — Diagnose-only agent swarms — ANTI-STRATEGY

Status: failed
Last updated: 2026-11-10

Fleet design is covered in depth by AGENT_FLEET.md (docs/). This file records only the evidence for "don't".

## When to use
Don't spawn many small agents that may not edit or score. Give each agent an owned outcome, a staged metric that can
see its effect (S11), a numeric target and a fixed suite.

## Preconditions & confirming probes
If you delegate diagnosis anyway: the child must be able to run probes with enough samples (S09) and its result must be
checkable by a number; share context in one file instead of each child rebuilding it.

## Procedure
See AGENT_FLEET.md.

## Verification
Count accepted candidates per agent-hour.

## Expected effect & cost
- 27 Haiku children (mean 8.7 min, 35 tool calls; 235 min total): **0 of 18 reviewed tracks accepted**, 0/27 led to a
  score improvement. Two returned while their job still ran; several had sign/axis errors.
- Confounded: children were Haiku, forbidden by brief to edit or score, working for a parent stuck on a plateau. A
  read-only doc agent in the old project did produce a research lead (r − ŷ decodable).

## Failure modes / interactions with other strategies
- Each child re-derives the same context (every child rebuilt and re-verified the 88,332-value recipe).
- Probes too small to resolve weak effects (4–8 frames at ±10 mm).

## Evidence log
- 2026-11-05, Codex line, 27 Haiku children. Evidence: CODEX/sparse-resume-20261105/haiku/parent_review.json; scratchpad phase01.md §2; scratchpad review/REVIEW.md §1.1.
