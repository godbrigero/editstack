# How to update the strategy library

Do this at the end of every run that used, tested or refuted a strategy, in the same turn you record the result on
the board. Keep edits small; the library is shared by concurrent agents.

## After a run
1. Open the strategy file you used. **Append** one bullet to `Evidence log` (never edit or delete old bullets):
   `- YYYY-MM-DD, <stack/model>, <variant>: <result with numbers, seeds, suite>. Evidence: <run dir / script / report path>.`
   Include negative and null results. If a number in an older bullet turns out wrong, append a bullet that says
   `CORRECTS <date> bullet: …` instead of changing it.
2. If you learned something that changes how to execute the strategy (a new precondition, a better default, a failure
   mode), edit that section and set `Last updated:` to today. Do not rewrite the history in the evidence log.
3. Update the row in INDEX.md if the status, typical gain or cost changed.

## When to change Status
- untested idea → **partial**: first run where the mechanism is verified offline or closed loop.
- partial → **proven**: a closed-loop gain beyond seed noise (qualification rules, ≥3 seeds on the target suite, paired
  vs parent), on at least one stack, with the exact procedure in the file.
- any → **failed**: worse or null on a fair test (correct harness, metric that can see the stage, ≥2 seeds). Say in the
  file what would have to be different to retry it.
- proven → partial: a later run on another stack fails to reproduce; keep both bullets.
Write the reason for every status change as an evidence bullet.

## Adding a new strategy
1. Take the next free id (highest `S<NN>` in this folder + 1; never reuse an id, even of a deleted or superseded file).
2. Create `S<NN>-<slug>.md` with the exact template: `Status:`, `Last updated:`, `When to use`,
   `Preconditions & confirming probes`, `Procedure`, `Verification`, `Expected effect & cost`,
   `Failure modes / interactions with other strategies`, `Evidence log`.
3. Add its row to the INDEX table and, if it answers a symptom, a line in the decision flow.
4. A new idea with no run is `untested idea`; it still needs a concrete procedure with the editstack ops/fit calls.

## Superseding a strategy
- Keep the old file. Change its first line under the title to `Superseded by S<NN> (<date>): <one-line reason>`,
  keep its Status as it was, and append an evidence bullet explaining the supersession.
- In INDEX.md, mark the row `superseded by S<NN>` in the "pick it when" column.

## Corrections to other docs
If your evidence contradicts a number in another doc (LESSONS.md, APPROACH_PLAYBOOK.md, …), add an item to
"Corrections to other docs" in INDEX.md with the doc, the claim, the correct value and the evidence path.
