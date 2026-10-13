# LLM Wiki Memory — Incremental Implementation Plan

**Status:** proposed implementation sequence. Nothing below is marked implemented solely because it appears in this plan.
**Delivery rule:** one small milestone per approved change. Report real verification, including anything not verified.
**Testing decision:** automated test authoring is deferred until the owner explicitly asks for it. Use non-destructive manual CLI/syntax checks in the meantime.

## Milestone 0 — Documentation and repository hygiene (now)

**Goal:** establish a stable project contract for Claude Code/Codex without introducing application code.

Deliver:
- Root `AGENTS.md` (shared instructions), `CLAUDE.md` (imports `AGENTS.md`).
- `docs/ARCHITECTURE.md` and this plan; a short README linking to them.
- Check `.gitignore` for editor files, venvs, credentials, and local corpus/runtime artifacts before creating those artifacts.

Done when: docs agree with the handoff and repository reality; unresolved external contracts are labeled unverified. No code, new dependencies, or infrastructure needed.

## Milestone 1 — First vertical slice: one synthetic document

**Goal:** show source-grounded atomic claims and a *proposal preview*, without writing Wiki files or Git commits.

Deliver only as needed:
- A minimal Python project/CLI and dependency manifest, chosen for the actually available Python version.
- One explicitly fictional Russian-language legal-like document as a local fixture.
- Small validated data models for available source identity, atomic claims, direct source quotes, and proposal actions.
- A thin, configurable client for an **approved local** OpenAI-compatible LLM endpoint; no coupling to `ai-assistant` users/billing/DB.
- Extractor that requests structured atomic claims (not only document summaries).
- Deterministic quote/source validation and an inspectable `--dry-run` preview. Invalid output fails closed.

Acceptance (manual): CLI processes the fixture, shows claims linked to exact source text and a readable proposal; a nonmatching quotation is rejected; the published Wiki is untouched. Record commands run, output, and missing checks.

Do **not** implement qmd, Git writer, online API, or a comprehensive package hierarchy here.

## Milestone 2 — Knowledge Consolidation

**Goal:** resolve where claims belong and generate constrained updates without mutating canonical Wiki.

- Add a minimal current-batch state and existing-page candidate interface (in-memory/static first).
- Add a typed consolidator action and independent semantic relation.
- Separate actual source-ID deduplication from semantic equivalence checks.
- Stage proposals with explicit `REVIEW_REQUIRED` on ambiguity/conflicts.
- Produce a deterministic preview of proposed page changes; reject bad IDs, broken references, unverified quotes, and illegal paths.

Acceptance (manual): demonstrate `CREATE_PAGE`, `UPDATE_PAGE`, `ADD_SOURCE`, `NO_CHANGE`, `REVIEW_REQUIRED` with synthetic examples. Two claims in one batch must see each other's staged state; repeating identical input must not produce new changes.

## Milestone 3 — Markdown Wiki and local Git publication

**Goal:** publish only validated/approved changes, with clear recovery semantics.

- Agree compact Markdown/frontmatter and stable claim/page identifier formats.
- Implement a single sequential writer, review gate, safe path generation, and strict staging.
- Start with a local **separate** Wiki Git repository; do not introduce a server unless needed.
- Add an appropriately small durable journal/checkpoint for retries and idempotency.
- Distinguish local commit, optional push, and later index stages.

Acceptance (manual): approved batch yields one intended change/commit; repeat does nothing; simulated interruption is recoverable without duplicate writes; unreviewed results cannot publish as verified.

## Milestone 4 — qmd search and index coordination

**Goal:** use qmd to find candidate Wiki pages and record the published/indexed snapshot boundary.

- Confirm qmd version, local model/download policy, CLI output, and Russian-language retrieval behavior.
- Integrate qmd candidate top-K search after exact and current-batch candidates.
- Track published Git SHA versus indexed Git SHA; never silently serve an out-of-date index as current.
- Obsidian remains optional and read-only relative to the canonical branch.

Acceptance (manual): added/updated Wiki pages are retrievable after indexing; restart/re-index is repeatable; a stale index is observable. No Elasticsearch replacement.

## Milestone 5 — Real source adapter (read-only)

**Goal:** process a small explicitly approved subset of real documents, if authorized.

- Verify relevant `ai-assistant` branch/revision and actual documents/headers/sections/revisions contracts.
- Read source data through a narrow, permission-controlled, read-only adapter.
- Preserve source identity and accessible citation offsets; mark missing revision/scope info unknown.
- Confirm model endpoint, resource limits, logging/data residency, and rights to process documents.

Acceptance (manual): known permitted source can be traced back to exactly the available original; no writes to source DB, no unauthorised data copying, no invented revision IDs.

## Milestone 6 — Offline retrieval evaluation

**Goal:** test the hypothesis before considering online integration.

- Freeze an approved source snapshot, Wiki Git SHA, prompts/model versions, and query set.
- Build an independently reviewed set of questions and target **original** document provisions.
- Compare existing ES+pgvector against Wiki-assisted retrieval using comparable `Recall@K`, `Precision@K`, error analysis, latency, and operational cost.
- Separate gains from simple summaries from gains due to cross-document consolidation.

Acceptance: an inspectable report with actual measurements, limitations, and a go/no-go recommendation. No claims of improvements without measured evidence.

## Later (not part of approved PoC) — ai-assistant integration

Only after demonstrated value and explicit approval: add a feature-flagged, authorized lookup path in `ai-assistant`; map Wiki hits to original document provisions; preserve current task isolation and fallback behavior; evaluate end-to-end answer quality. No silent switch from original-source-based responses.

## Current next action

Approve this documentation set, adjust any disputed architecture decisions, and incorporate it into the checked-out `master` branch. Then design **Milestone 1 only** before generating its implementation code.
