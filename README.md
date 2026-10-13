# LLM Wiki Memory

An **offline-first proof of concept** for building a source-grounded, Git-versioned thematic Wiki from legal documents and evaluating whether it improves retrieval of original legal provisions.

**Status:** design/documentation stage. No Wiki Builder, search index, or integration is implemented yet.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system boundaries, data flow, invariants, and open questions.
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md) — small, sequential milestones and acceptance criteria.
- [Agent instructions](AGENTS.md) — shared code-quality, safety, and development rules.
- [Claude Code instructions](CLAUDE.md) — imports shared agent instructions.

## Guiding decisions

- Original legal documents remain authoritative; the Wiki is derived navigation and retrieval material.
- Published Wiki: Markdown + a separate Git repository; one sequential writer.
- Proposed Wiki search: qmd; existing `ai-assistant` ES/pgvector remains unchanged during the PoC.
- LLMs propose changes; deterministic validation and review control publication.
- Initial work uses synthetic documents, not confidential or corporate source material.

## Setup

Runtime, dependencies, and CLI will be documented after the first implemented milestone. There are no verified install/run commands yet.
