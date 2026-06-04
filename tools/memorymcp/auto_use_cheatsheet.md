# memorymcp — cheatsheet

Treat memorymcp as a reflex, not a lookup DB.

**Contract**
- Session start → `queryMemory` for project / agent context.
- Before any non-trivial task → `queryMemory` for prior art.
- After file_open / file_edit / test_run / commit / discovery → `onAgentAction` (one-liner).
- End of task → `upsertMemory` for any pattern, decision, trick, or lesson.
- Related to existing memory → `createMemoryEdge` (relation: refines / depends_on / follows / contradicts / example_of).

**End-of-turn checklist:** `query ✓   store ✓   link ✓   self-check done ✓`

**Redact first if uncertain:** `redactSensitive` → then `upsertMemory`.

**Retention:** `permanent` for decisions / architecture / hard lessons; `temp` for in-flight; `auto-delete` (default) for transient.

**Anti-patterns:** do NOT skip query for small tasks, do NOT store secrets (redact first), do NOT store without a `memory_type` + tag, do NOT store generic LLM filler — store the non-obvious part.

**Long docs:** run `textToGraph` once (output="adjacency") before reasoning over them.

**Two levels of analysis** — tag every memory with `level:meta` (1-3 big design calls per topic, e.g. "why unified launcher process") or `level:detail` (many concrete patterns per topic, e.g. "token bucket refill formula"). For "why was X built this way?" use `getMetaDecisions(query)`. For "how do I do X?" use `queryMemory(query)`. Store the big call as `level:meta`, then store each of its patterns as `level:detail` — link them with `refines` so the graph is meta→detail.

**Priority** — `priority:A` (must-know, ~1-2/topic), `priority:B` (should-know, default for patterns), `priority:C` (nice-to-know, edge cases). Tag every memory with the right priority. The three axes (memory_type × level × priority) compose: `code_pattern + level:detail + priority:A` is a must-know concrete pattern. When context is tight, query only `priority:A` via `queryMemory(query, tags=["priority:A"])` or `getMetaDecisions(query)` (which defaults to A).

**Full policy:** call `getMemoryAutousePolicy()`.
