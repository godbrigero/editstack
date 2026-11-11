# Orchestration: one-page summary for the lead

The full, evidence-backed guide is **AGENT_FLEET.md** (roles, model tiers, counts, behaviour contracts, briefs,
lifecycle, costs). This page is the checklist. Numbers come from LESSONS.md.

1. **Phases 0-2 alone.** One agent holding the whole context does the harness audit, structural fixes and first
   analysis. Start the baseline runs detached right away; they run while you read.
2. **Delegate from Phase 3,** once there is a causal sentence with a number. Split by measurable quantity: a stage
   (reach / grip / cameras) or a mechanism. Every agent gets a staged metric that can SEE its effect, a numeric
   target, a fixed suite and its own folder. Never give two agents the same quantity or hypothesis.
3. **How many:** `N_research = floor(min(P_ram, P_sat) / 2)`, at 2 model processes each.
   `P_ram = floor((RAM_budget - 9) / per_process_gb)`; P_sat is where episodes per hour stop rising. That is 2 on a
   60 GB machine with ~9 GB processes, 3 at most. Nothing else that loads models may share the pool.
4. **Tiers:** frontier models for the lead, stage owners and reviewers. Mid tier for doc/render. Fast tier only for
   jobs a checker can verify (reproduction, arithmetic audits), never for interpretation. Details and evidence:
   AGENT_FLEET.md.
5. **Brief** with `editstack agent new` plus the causal sentence, what was already tried, and the default to take
   at each choice. State defaults and proceed. Don't wait for go-aheads: that applies to the lead too.
6. **While they run:** no polling. Use completion notices and `editstack status`. Don't run your own variants on an
   agent's line. Watch CLAIMS.json and MODEL_MAP.md for overlap. After network drops: "resume, don't relaunch".
7. **Merge:** one chain (PARENT); re-qualify after each link; check for double correction; run `editstack ablate`
   before promoting.
8. **Verify:** start an independent reviewer (fresh context, given the code and the evidence, not your conclusions)
   before promoting or publishing numbers. In this project one caught 5+ documentation errors and 4 critical tool bugs.
9. **Doc / render work** goes to a separate agent with one write folder. Keep it out of the lead's context.
10. **Record:** every experiment ends with `editstack learn`. See AGENTS.md Phase 6.
