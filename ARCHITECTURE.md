# LLM Wiki Memory — Architecture

**Status:** PoC design agreed in discussion; components described here are *planned*, not implemented.
**Baseline:** 2026-10-13 design handoff. Revalidate assumptions against the real `ai-assistant` branch and available source contracts before integrating.

## 1. Problem and objective

The existing legal assistant uses Elasticsearch plus PostgreSQL/pgvector to find original documents, then a local LLM builds answers. Large acts, cross-document references, exceptions, and revisions can make retrieval incomplete. The **unproven hypothesis** is that a source-grounded thematic Wiki can reveal additional relevant original legal provisions.

LLM Wiki is derived knowledge, not legal authority or a replacement for the source corpus. The offline PoC must measure whether Wiki-assisted retrieval helps, without altering production RAG.

## 2. Scope and explicit non-goals

**In scope:** offline extraction of atomic claims; source-grounded provenance; candidate page resolution; constrained consolidation proposals; deterministic validation; inspectable dry-run; Markdown/Git publication after review; qmd retrieval; later controlled comparison with baseline retrieval.

**Out of scope for the first PoC:** rewriting `ai-assistant`, serving end users, full legal-review workflow, automatic legal verification, a graph database, Kafka/Redis, new canonical SQL storage, storing original PDF/DOCX in Wiki Git, multiple concurrent writers, and general-purpose agent frameworks.

## 3. Logical flow

```text
Approved synthetic inputs (initially) / existing source corpus (later, read-only)
    -> Source adapter: exact text + available document/revision identifiers
    -> Section/claim preparation: preserve headings, scope, and citation offsets
    -> LLM extractor: atomic claim proposals + direct source quotations
    -> Deterministic source/quote validation
    -> Candidate resolver: exact IDs/aliases + current-batch changes + later qmd top-K
    -> LLM consolidator: action + semantic relation + proposed edit
    -> Proposal validator: provenance, scope, IDs, references, safe paths
    -> Staging / dry-run (no published Wiki writes)
    -> [explicit approval/review] one sequential Git writer
    -> Wiki Markdown Git commit/push -> qmd update/embed -> index-ready snapshot
    -> Offline retrieval evaluation against existing ES + pgvector
    -> [only with evidence and approval] opt-in ai-assistant integration
```

These are **logical responsibilities**, not a mandatory one-class/one-file or microservice-per-step design.

## 4. Storage and ownership

| Component | Role | Canonical? |
| --- | --- | --- |
| Original legal document store | Original text/structure/revisions, when available; read-only to this project | Canonical for legal source text |
| Wiki Markdown Git repository | Published thematic pages, stable claims, references, and revision history of derived content | Canonical for **published Wiki** |
| Local staging/progress journal | Proposed changes, review state, batch/publish checkpoints; recoverable | Operational state, not published legal knowledge |
| qmd (local SQLite index) | Full-text/semantic Wiki retrieval; rebuilt from a published Git snapshot | No |
| Obsidian (optional) | Human browsing of Markdown and links | No |

The Wiki Git repository is distinct from the application code repository. Keep source documents and confidential Wiki data in an approved private environment; never use a public repository for restricted data.

## 5. Domain contracts to refine before implementation

Proposed conceptual records (not existing Python APIs or confirmed source-DB columns):

- `Source`: stable `document_id`, full text, and **optional** revision/section/span data only if the source actually provides it.
- `Claim`: stable ID, atomic statement, verbatim supporting quote, source identity, conditions, exceptions, scope, and known time/jurisdiction fields.
- `WikiPage`: stable page ID, title, aliases, linked claims, sources, related page IDs, publication/review status.
- `Proposal`: action, target page (if applicable), candidate claim, relation, rationale, and `requires_review` flag.
- `Batch`: stable input identity, stage, expected Git base SHA, publication SHA and indexing SHA when relevant.

Consolidation actions:

| Action | Meaning |
| --- | --- |
| `CREATE_PAGE` | Candidate topic has no suitable existing page; stage a new one |
| `UPDATE_PAGE` | Add an independent rule, qualification, or exception to an existing page |
| `ADD_SOURCE` | Add provenance for a genuinely equivalent rule; no duplicate claim text |
| `NO_CHANGE` | Already-processed input/revision/extraction gives no effective change |
| `REVIEW_REQUIRED` | Ambiguous, conflicting, unsupported, or invalid proposal; do not publish |

Semantic relations (separate from actions): `SAME_RULE`, `NEW_RULE`, `NARROWS_SCOPE`, `ADDS_EXCEPTION`, `POTENTIAL_CONFLICT`, `RELATED_TOPIC`, `UNKNOWN`.

Stable IDs must not depend solely on editable titles. A quote hash alone does not prove legal equivalence. Exact JSON/Pydantic schema, ID format, and Markdown frontmatter are **not yet finalized**; agree them in the corresponding small implementation milestone.

## 6. Resolution and consolidation

Resolve candidates in this order:

1. Stable IDs, source references, canonical topics, aliases.
2. In-memory changes already proposed in the **current batch** (qmd may be stale).
3. qmd retrieval of nearby published Wiki pages (when integrated), followed by loading the full candidate contents.
4. Propose creation only after checking for duplicate IDs/topics; uncertainty goes to review.

The LLM decides neither filesystem paths nor publication. It outputs a typed proposal, which application code validates. Do not merge similar rules merely by embedding similarity. Preserve exceptions, effective dates, jurisdiction, and the full original conditions.

## 7. Provenance and legal safety

- A published claim must trace to a real source and exact quote; where revision/section identifiers do not exist, record that limitation rather than inventing values.
- Verify the supplied quote against the appropriate source snapshot. An unverifiable claim cannot be auto-published as `verified`.
- LLM output does not establish that two legal norms are equivalent, supersede one another, or apply on a particular date.
- `needs_review`/`draft` content must not appear as approved legal fact. Promotion to `verified` requires a defined review process, outside initial PoC.
- Original-source authorization and access controls must be preserved in any later integration; a Wiki link never grants permission to see a document.

## 8. Git publication and index consistency (later milestone)

One writer processes approved batches sequentially. For each batch: check a clean/expected worktree; stage only its files; commit once; push without force; index that exact published snapshot; record progress.

A local Git commit, remote push, and qmd indexing are **not one transaction**. Track a published commit identity separately from indexed commit identity. A retry after commit or push must resume, not duplicate already-applied claims or commits; unexpected remote changes require an explicit stop/reconciliation. Never conflate Git Markdown history with effective versions of legal acts.

For the initial CLI, `--dry-run` must not mutate published Wiki files or perform commit/push/index operations. It may create an **untracked local report outside the Wiki** only if explicitly requested, and should otherwise print proposals to stdout.

## 9. External integration boundaries

`ai-assistant` is a separate codebase and stays unchanged until the offline comparison. A future adapter would map Wiki hits back to authorized, original `document_id` plus specific provision/revision when available, rather than answering from the Wiki summary alone. Elasticsearch scores, pgvector similarity, and qmd scores are not directly interchangeable.

Before implementing read-only ingestion, inspect the **actual** `ai-assistant` source schema, importer, current branch, access controls, and available model endpoint. The existence of `documents`, `headers`, and `sections` in the inspected code does **not** confirm reliable source revision IDs or legal paragraph identities in deployed data.

## 10. Decisions and questions

**Agreed:** Python; offline-first incremental PoC; Markdown + Git published Wiki; one writer; qmd for Wiki search; Obsidian optional; existing RAG preserved; constrained LLM proposals; review-on-uncertainty.

**Not yet confirmed:** actual DB source/revision contract, whether allowed corp source data can be used, local LLM capabilities and JSON mode, qmd/runtime availability and Russian legal retrieval quality, canonical claim/page schema, reviewer ownership, production integration conditions.

Do not turn unresolved questions into implicit implementation assumptions.
