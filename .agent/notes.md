# NOTES.md — Evolving Decisions

Captured by `/molexp-note`. Stable entries are promoted to CLAUDE.md then deleted here.

<!-- Format: ## <slug> | <date> | <author>
Decision body.
**Status:** evolving | stable | promoted -->

## cross-layer-data-reference | 2026-05-06 | impl

Cross-layer references go through URI / asset_id / string id, never `from <upstream_layer> import SomeType` to define a shape in `<downstream_layer>`. If two layers need the same type, move the type to the downstream common layer (or a shared root); duplicating it in the upper layer is the bug pattern. **Counter-example**: `molexp.workflow.proposal` once imported `molexp.agent.types.ArtifactRef` — corrected to `code_artifact: Path | None`.

Pure data types → `pydantic.BaseModel(frozen=True)`. Runtime containers (live callables / asyncio objects / non-pydantic instances) → plain Python class with explicit `__init__`. **`arbitrary_types_allowed=True` is forbidden in `src/molexp/agent/`** — anything that needs it is a runtime container by definition. See full table in `CLAUDE.md ## Data type ownership`.

**Status:** stable
