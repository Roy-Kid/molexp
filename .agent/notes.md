# NOTES.md — Evolving Decisions

Captured by `/molexp-note`. Stable entries are promoted to CLAUDE.md then deleted here.

<!-- Format: ## <slug> | <date> | <author>
Decision body.
**Status:** evolving | stable | promoted -->

## cross-layer-data-reference | 2026-05-06 | impl

Cross-layer references go through URI / asset_id / string id, never `from <upstream_layer> import SomeType` to define a shape in `<downstream_layer>`. If two layers need the same type, move the type to the downstream common layer (or a shared root); duplicating it in the upper layer is the bug pattern. **Counter-example**: `molexp.workflow.proposal` once imported `molexp.agent.types.ArtifactRef` — corrected to `code_artifact: Path | None`.

Pure data types → `pydantic.BaseModel(frozen=True)`. Runtime containers (live callables / asyncio objects / non-pydantic instances) → plain Python class with explicit `__init__`. **`arbitrary_types_allowed=True` is forbidden in `src/molexp/agent/`** — anything that needs it is a runtime container by definition. See full table in `CLAUDE.md ## Data type ownership`.

**Status:** stable

## approval-gate-instance-name | 2026-06-21 | impl

`ApprovalGate` takes an optional `name=` kwarg that overrides the stage's
ledger name on that **instance only** (set via `object.__setattr__(self,
"name", name)` in `__init__`; class-level `ApprovalGate.name` stays
`"approval_gate"`). Reason: a harness `Mode` keys its per-run completion
ledger on `stage.name`, and `stage_fingerprint()` keys on the **class** alone
(instance config is excluded) — so two same-named `ApprovalGate`s in one mode
would make the second a false ledger cache-hit and get silently skipped.
`PlanMode` wires two gates: the early experiment-report review gate is named
`approve_experiment_spec`, the terminal final-report gate keeps the default
`approval_gate`.

**Rule**: when a `Mode` wires more than one instance of the same `Stage`
class, give the extra instances a distinct `name=` so each gets its own
completion-ledger key.

**Status:** stable

## three-layer-rectification | 2026-05-09 | impl

The molexp dependency DAG was inverted by the rectification spec:
**workspace ← workflow ← agent**, where the arrows point from "uses" to "is used by".

Why the inversion: pre-2026-05-09 the codebase had workspace importing
workflow types (`Experiment.workflow: WorkflowSpec`,
`workspace/sessions.py:SessionLibrary` knowing about agent session
schemas, `workspace/subsystem.py` hardcoding `"agent.sessions"`),
which made workspace neither a clean storage primitive nor a clean
upstream concern. The fix:

- workspace becomes a pure storage primitive (filesystem hierarchy,
  atomic JSON, content-addressed assets, generic per-kind `SubsystemStore`).
  Knows nothing about workflows, sessions, agents, or LLMs.
- workflow becomes a graph engine that uses workspace for caching
  (`WorkspaceCacheStore` backed by `SubsystemStore("workflow.cache")`)
  and for atomic state writes (`workspace.atomic_write_json` for
  `workflow.json` snapshots). The `~/.molexp/cache/` user-home
  shortcut is gone.
- agent becomes a thin LLM harness that uses both downstream layers
  through their public surfaces, and confines `pydantic_ai` to
  `agent/_pydanticai/` (lazy load) and never imports `pydantic_graph`
  at all.

Mechanical enforcement: three import-guard tests
(`tests/test_<layer>/test_import_guard.py`). The audit + design lives
in `.claude/specs/molexp-rectification.md`; the binding "done"
contract is `molexp-rectification.acceptance.md` (criteria
P0-01..P0-07, P1-01..P1-05, P2-01..P2-04, P3-01..P3-05, P4-01..P4-02,
P5-01..P5-02, P6-01..P6-03).

Side effects worth remembering:
- `Experiment.workflow` field, `Experiment.set_workflow`, the
  workspace-side `_promote_to_workflow` / `_resolve_*_entrypoint`
  helpers, and `workspace/sessions.py:SessionLibrary` are **gone**.
  Pairing an Experiment with a workflow is the caller's concern;
  workflow exposes `promote_callable` / `WorkflowSnapshotRef`
  publicly.
- `agent/_legacy_types.py` is gone; `ToolSchema` / `ModelToolCall`
  live permanently in `agent/tools/spec.py`; `to_jsonable` lives
  privately in `agent/sessions/_serde.py`.
- Old on-disk `experiment.json` files with a `workflow` field still
  load (`ExperimentMetadata` carries `extra="ignore"`).

**Status:** stable
