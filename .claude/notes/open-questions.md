# Open questions

Open questions that downstream sub-specs in the
`planmode-workspace-pipeline-*` chain (sub-specs 01–06) must
**accept the recommended default** or **override with explicit
justification** before starting substantive work.

Each question is paired with the recommended-default answer. A
sub-spec that overrides a default does so in its own design
section, not by editing this file.

## 1. Where do PlanMode-materialized experiment workspaces live on disk?

Candidates considered:

- **(A) `Workspace.subsystem_store("agent.plan-experiments")` —
  RECOMMENDED DEFAULT.** Each plan instance lives at
  `<workspace_root>/.subsystems/agent.plan-experiments/<plan_id>/`.
- (B) Under a specific `Experiment` / `Run` directory (e.g.
  `runs/<run-id>/plan/`).
- (C) A new generic `workspace/` primitive (e.g. an
  `ExperimentWorkspace` sibling of `Run`).

**Recommended default rationale.** Option A respects the workspace
charter rule that *"subsystem identifiers are owned by consumers"*
(CLAUDE.md `## Layer charters`), requires zero changes to
`workspace/` source, and leaves the door open for sibling
subsystems (e.g. `agent.run-experiments`) without requiring a new
primitive each time. Option B couples the plan workspace to a
specific run, which is wrong polarity — a plan can be authored
*before* any run exists. Option C is over-engineering until a
second consumer asks for the same shape.

**Reservation status.** `agent.plan-experiments` is reserved in
the architecture blueprint's Conventions section (sub-spec
00, 2026-05-09). Sibling kinds like `agent.run-experiments` are
**not** reserved at this time — reserve when a sub-spec needs them.

**Adopted by:** sub-spec 03 (`planmode-workspace-pipeline-03-experiment-workspace-layout`).

## 2. Should `ValidationReport` be a generic `ValidationAsset` subtype, or remain a plain markdown file inside the experiment workspace?

Candidates considered:

- **(A) Plain `validation_report.md` file inside the experiment
  workspace, written via `PlanWorkspaceHandle.write_validation_report` — RECOMMENDED DEFAULT for v1.**
- (B) New generic `ValidationAsset` subtype under
  `molexp.workspace.assets`, registered with the asset catalog so
  cross-cutting consumers can query for it.

**Recommended default rationale.** A `ValidationAsset` is only
worth the cost when there is a second consumer that wants to query
for "all validation reports across all experiments". v1 of the
PlanMode pipeline has exactly one consumer (the human reviewer
reading the markdown), so we keep it as a file. If a future
RunMode (or a UI dashboard) wants cross-cutting visibility into
validation failures across plans, **revisit this decision** by
opening a new spec that introduces `ValidationAsset` and migrates
existing `validation_report.md` files into the catalog.

**Adopted by:** sub-spec 03 (file-only) and sub-spec 06 (consumer).

## 3. What is the on-disk + in-process shape of the `PlanMode → RunMode` handoff contract?

Candidates considered:

- **(A) Frozen pydantic type, serialized to a `handoff:` section
  inside `manifest.yaml` — RECOMMENDED DEFAULT.** In-process
  consumers read `AgentRunResult.mode_state["plan"]["handoff"]`;
  on-disk consumers read `manifest.yaml`.
- (B) On-disk YAML manifest only (no in-process pydantic mirror).
- (C) Frozen pydantic type only (no on-disk persistence; consumer
  must keep the agent process alive).

**Recommended default rationale.** Option A combines static
enforcement (the pydantic type rejects malformed handoffs at the
boundary) with disk-side inspectability (a future RunMode can
load the contract without depending on the agent runtime). Option
B loses static enforcement; option C breaks the agent ↔ run
process boundary because the contract dies with the agent. The
serialization round-trip is plain
`yaml.safe_dump(json.loads(handoff.model_dump_json()))` — no
custom encoders, so the YAML form is always reconstructible by
calling `PlanRunHandoff.model_validate_json` on the JSON
intermediate.

**Adopted by:** sub-spec 06 (`planmode-workspace-pipeline-06-pipeline-rewrite-codegen-validation`)
defines `PlanRunHandoff` in `src/molexp/agent/modes/plan/handoff.py`.

## Integrity issues surfaced during the 2026-05-09 blueprint refresh

These are tracked here (rather than in `architecture.md`'s managed
section) because they are *transient* — they should be cleaned up
by the relevant sub-spec rather than recorded permanently.

- **Stale import in `src/molexp/agent/_pydanticai/provider.py`**
  imports `from molexp.agent.modes._plan_protocols import ModelTier`,
  but no `_plan_protocols` module exists on disk. The actual
  location is `molexp.agent.modes.plan.protocols`. **Fix in:**
  sub-spec 02 (already noted in the spec body).
- **Empty leftover plugin directories** at
  `src/molexp/plugins/agent_pydanticai/` and
  `src/molexp/plugins/metrics/`. No `__init__.py`, no modules.
  The agent plugin indirection was removed; metrics moved to
  `workspace.metrics`. **Recommended action:** delete both
  directories — out of scope for the PlanMode-rewrite chain;
  open a separate cleanup spec when convenient.
- **Empty `src/molexp/sweep/` package** with no `__init__.py`.
  Either reserved for a future spec or leftover; track outside
  this chain.
