# molexp Layer-Integration & AI-Assisted Operation Spec

**Status**: target architecture (coordination layer). Companion to
`harness-goal.md` (which defines *what the harness is* — provenance-first, agent
proposes / harness disposes) and `architecture.md` (which maps *current state*).
This file defines *how the existing layers cooperate* and *how the agent operates
over them*. It does **not** redefine Workspace / Workflow / Experiment / Run /
Artifact / Knowledge / AgentTask — those exist (see `architecture.md`). It adds
the **coordination contracts** between them. Reconcile incrementally; each slice
in §9 lands a piece.

> **Hard rule — no parallel architecture.** Everything below is expressed in
> terms of modules that already exist. New code is a thin *seam* (a read-model
> assembler, an event log, a proposal schema, a few stages), never a second copy
> of an existing subsystem. Where a concept already lives somewhere, this spec
> says *reuse it*, not *rebuild it*.

---

## 0. The coordination loop

The product thesis — *a scientific work system bounded by Workspace, structured
by Workflow / Experiment / Run, grounded by Artifact and Knowledge, operated
through Agent-assisted actions* — reduces to one loop the whole system must
support:

```text
WorkspaceContext                 (§1 — assembled from authoritative state)
  → UserIntent / AgentIntent     (§6 — structured, not free chat)
  → Plan                         (§7 — Experiment Planner = today's PlanMode)
  → ChangeProposal               (§8 — first-class, reviewable, gated)
  → Action                       (§6 — controlled executor / lifecycle op)
  → Run / Artifact               (§3, §4 — existing workspace + harness machinery)
  → KnowledgeDelta               (§5 — source-linked KnowledgeItems)
  → updated WorkspaceContext     (loop closes; §2 events drive the refresh)
```

Each arrow is a contract between two layers. The rest of this document specifies
those contracts and which existing module owns each side.

**Ownership stance carried throughout** (these resolve the gaps in
`architecture.md`'s audit):

- Canonical state is **workspace on disk** (entity `*.json`, `_ops/run.json`,
  `assets.json`, OKF `meta.yaml`/`index.md`). Agent memory and frontend state are
  **never** canonical.
- The agent **proposes**; the harness **disposes**. High-risk mutations go
  through a `ChangeProposal` + approval gate before any executor runs.
- Every artifact used in reasoning is **citeable by source reference**; every
  KnowledgeItem carries **source attribution**; every agent action is
  **traceable** through the event + provenance spine.

---

## 1. WorkspaceContext

### 1.1 What it is

`WorkspaceContext` is the **canonical read-model** the system hands to agents and
planners so they reason over structured state instead of scraped chat. It is a
*projection* assembled on demand from authoritative sources — it stores nothing
new and is never itself canonical.

Today three places independently re-assemble overlapping context:
`server/routes/workspace.py` (`GET /info|/runs|/files`), the UI
`WorkspaceSnapshot` (`ui/src/app/state/useWorkspaceState.ts`, polled every 3 s),
and ad-hoc prompt assembly inside the plan/agent runtimes. **This spec unifies
them behind one assembler** so all three become consumers of the same shape.

### 1.2 Shape (illustrative)

```python
class WorkspaceContext(BaseModel):
    # identity
    workspace: WorkspaceRef                      # id, name, root, targets
    # focus (ephemeral — supplied by the caller, NOT stored)
    focus: ContextFocus                          # active project/experiment/run + selection
    # structure
    projects: list[ProjectRef]
    experiments: list[ExperimentRef]             # incl. parameter_space summary
    workflows: list[WorkflowRef]                 # available compiled/IR workflows
    # execution
    recent_runs: list[RunRef]                    # ordered by finished/started
    failed_runs: list[RunRef]                    # status ∈ retryable domain
    running_runs: list[RunRef]
    # grounding
    artifacts: list[ArtifactRef]                 # recent / focus-scoped
    knowledge: list[KnowledgeRef]                # relevant items (§5)
    open_questions: list[KnowledgeRef]           # OpenQuestion items + spec ResolvedQuestion gaps
    # health
    stale_or_missing: list[HealthFlag]           # see §1.4
```

`ContextFocus` (active project/experiment/run + `selected_object_refs`) is the
**only** part that comes from the caller's ephemeral session/selection — it is
passed in, never persisted as workspace state. Everything else is derived from
disk.

### 1.3 Producers vs consumers

| Field group | Authoritative source (PROVIDER) | Module |
|---|---|---|
| workspace identity, projects, experiments | `Workspace` / `Project` / `Experiment` folder tree | `workspace/workspace.py`, `project.py`, `experiment.py` |
| workflows available | `Experiment.workflow.json` IR + compiled refs | `workspace/experiment.py:98`, `workflow/codec.py` |
| runs (recent/failed/running) | `Run` + hot-state sidecar | `workspace/run.py:225` (`read_ops`), `run_ops.py` |
| artifacts | manifest scan | `workspace/assets/scan.py` (`scan_assets`/`get_asset`) |
| knowledge / open questions | OKF concepts via Bundle | `workspace/bundle.py`, `concepts.py`, §5 |
| health flags | computed over the above | new assembler (§1.4) |

**New seam (P0):** a `workspace.context` read module (layer-legal: it imports
only `workspace` + `knowledge`, never `workflow`/`agent`/`harness`) that composes
the above into `WorkspaceContext`. It is a **pure read**, reusing `read_ops`,
`scan_assets`, and `Bundle`.

**Consumers (must only read it):** the server (`GET /context`, replacing the
piecemeal `workspace.py` reads), the CLI (`molexp context`), the UI
`WorkspaceSnapshot` (re-pointed at `/context`), and every agent feature in §7.
The agent-facing *filtered/ranked* view (relevance, token budget) is assembled
one layer up (harness/server) on top of the canonical context — workspace stays
relevance-agnostic.

### 1.4 Health flags (stale / missing / failed)

Computed, never stored:

- **failed run** — `Run.status ∈ RETRYABLE_STATUSES` (`workspace/run.py:61`).
- **stale running** — `running` with dead owner PID or stale heartbeat (reuse the
  existing reaper rule in `run_ops.py`).
- **missing output** — an `ExpectedOutput`/`acceptance_criteria` (harness
  `WorkflowIR`) with no matching `ArtifactAsset` in `scan_assets`.
- **stale knowledge** — KnowledgeItem whose `sources` point at a superseded
  artifact content-hash or a deleted run (§5.4).
- **orphan artifact** — asset with a `Producer.run_id` that no longer resolves.

These flags are exactly what the Workspace Copilot (§7.1) surfaces as "next
actions."

---

## 2. Cross-layer event model

### 2.1 Three existing streams + one new spine

The repo already has two event systems; this spec adds one coordination spine and
defines how they relate. **No stream is replaced.**

| Stream | Scope | Persisted? | Existing module | Role |
|---|---|---|---|---|
| `HarnessEvent` | one harness run | yes (`run_dir/harness.sqlite`) | `harness/store/sqlite_event_log.py` | deep per-run audit (stage/agent/approval/artifact) |
| `AgentEvent` | one agent loop | no (in-memory stream) | `agent/events.py` (`AsyncIteratorEventSink`) | live UI streaming (tokens, tool calls, thinking) |
| **`WorkspaceEvent`** *(new, P0)* | whole workspace | yes (`<root>/workspace.events.sqlite`) | mirror `SQLiteEventLog` | **cross-object coordination spine** |

The new `WorkspaceEvent` log is the canonical "what happened across objects"
timeline — append-only, `seq`-ordered, never overwritten, at workspace scope.
Per-run detail stays in `harness.sqlite`; the workspace log holds the coordination
event + a `run_id`/`content_hash` pointer to drill down. This keeps the deep audit
local to a run while giving planners/copilot one place to observe the whole
project.

> **OPEN DECISION (blocks P0.3 — not P0.1).** The proven append-only SQLite
> primitive lives in `harness/store/_sqlite.py`, but `workspace` **cannot import
> `harness`** (layer DAG). Literally "mirroring" it into `workspace/events.py`
> would be a **second copy of an existing store — which invariant #1 (§10)
> forbids.** Resolution: **extract the shared append-only-SQLite primitive to a
> Layer-0 module** (a peer of `molexp.atomicio` / `molexp.ids`) that BOTH
> `harness.store` and `workspace.events` cite — one implementation, two scopes.
> Decide this before P0.3; do not copy the store.

`workspace.selection.changed` is **ephemeral** — it stays a UI/session signal
(reuse the existing `ui/src/app/state/workspaceSwitchEvents.ts` CustomEvent bus
pattern) and is **never** persisted or given provenance.

### 2.2 Event catalogue

P = produces provenance edge · A = agent-visible · K = may trigger knowledge
extraction.

| Event | Producer | Consumers | Payload (core) | P | A | K |
|---|---|---|---|---|---|---|
| `workspace.asset.added` | `ArtifactAccessor.save` / asset register (`workspace/assets/accessors.py`) | Context, Artifact Interpreter, UI | asset_id, scope, kind, content_hash, producer(run/exec/task) | ✓ | ✓ | ✓ |
| `workspace.selection.changed` | UI / session | agent focus view | selected_object_refs | ✗ | ✓ | ✗ |
| `workflow.created` | `Experiment.set_workflow` / compile (`workflow/compiler.py`) | Context, Planner | workflow_id, version, ir_hash | ✓ | ✓ | ✗ |
| `workflow.validated` | WorkflowIR/Bound validators (`harness/validators/`) | Planner, Copilot | workflow_id, ok, review_flags | ✓ | ✓ | ✓ (on fail) |
| `workflow.changed` | applied WorkflowChangeProposal (§8) | Context, runs | workflow_id, diff_ref, proposal_id | ✓ | ✓ | ✓ |
| `experiment.created` | `Project.add_experiment` | Context, Planner | experiment_id, project_id, parameter_space | ✓ | ✓ | ✗ |
| `experiment.plan.updated` | `materialize_plan_records` (`server/plan_runtime/materialize.py`) | Context, UI | experiment_id, spec/plan artifact refs | ✓ | ✓ | ✓ |
| `run.created` | `Experiment.add_run` (`workspace/experiment.py:376`) | Context, Monitor | run_id, experiment_id, params, config_hash | ✓ | ✓ | ✗ |
| `run.started` | `RunLifecycle.enter` (`workspace/run_lifecycle.py:52`) | Monitor, Context | run_id, exec_id, owner, started_at | ✓ | ✓ | ✗ |
| `run.failed` | `RunLifecycle.exit` / reaper | Run Monitor → FailureAnalysis, Context | run_id, exec_id, error, last_stage | ✓ | ✓ | ✓ |
| `run.completed` | `RunLifecycle.exit` | Monitor, Artifact Interpreter, Context | run_id, exec_id, output_refs, finished_at | ✓ | ✓ | ✓ |
| `artifact.created` | `ArtifactAccessor` / harness `FileArtifactStore` | Interpreter, Context, provenance | ArtifactRef (id, kind, content_hash, parent_ids) | ✓ | ✓ | ✓ |
| `artifact.updated` | re-register / new version | Interpreter, Context | new ref + supersedes_id | ✓ | ✓ | ◻ |
| `knowledge.created` | KnowledgeItem write (§5) | Planner, Copilot, Context | item_id, kind, sources, status | ✓ | ✓ | ✗ |
| `knowledge.conflict.detected` | Knowledge Curator (§7.6) | Curator, UI, user | conflicting_ids, basis | ✓ | ✓ | → proposal |
| `agent.intent.received` | server/CLI entry (`agent_tasks`, `plan_tasks`, `curate_tasks`) | agent loop / mode | intent_text, mode, focus, actor | ✓ | ✓ | ✗ |
| `agent.proposal.created` | agent loop / harness mode (§8) | ApprovalGate, UI review, user | proposal_id, intent, affected_objects, risk | ✓ | ✓ | ✗ |
| `agent.action.completed` | executor / controlled op after approval | Context, Monitor, provenance | action_id, proposal_id, result_refs, status | ✓ | ✓ | ✓ |

### 2.3 Provenance contract

A "P ✓" event MUST write a provenance edge into the lineage spine when it links
artifacts: reuse `harness/store/sqlite_lineage_store.py` (`add_edge(parent,
child, relation, stage, run_id)`) for in-run derivation, and the workspace
`Producer` field (`workspace/assets/base.py`) for run→artifact lineage. Knowledge
links reuse the typed OKF out-edge (§4.3 / §5.2). The three together answer
"trace this artifact back to the user plan" (the harness-goal §17.6 acceptance).

---

## 3. Workflow / Experiment / Run cooperation

This is **lifecycle + data flow only** — the entities are unchanged.

```text
Experiment ──references──► Workflow (IR)         Experiment.workflow.json  (workspace/experiment.py:98)
    │                          │
    │ add_run(params|space)    │ compile()                                 (workflow/compiler.py)
    ▼                          ▼
   Run ──binds──► (workflow × params × env × time)                         (workspace/run.py, RunContext)
    │ execute
    ▼
 Execution ──produces──► ArtifactAssets  ───► (interpret) ───► KnowledgeItems   (§4, §5)
    │                                                              │
    └────────────────── run.failed ──► FailureAnalysis ◄──────────┘
                                                                   │
 Knowledge ──informs──► next ExperimentPlan / WorkflowChangeProposal (§7, §8)
```

### 3.1 Experiment ↔ Workflow

- An `Experiment` **references** a workflow, it does not embed engine code. The
  reference is the externalized IR doc `Experiment.workflow.json`
  (`workspace/experiment.py:98`) + a compiled hash. **Reconciliation needed**
  (architecture audit gap): make the *workflow-layer IR* (`workflow/codec.py`,
  `schema/workflow.json`) the single canonical workflow definition; the harness
  intent IR (`harness/schemas/workflow_ir.py`) and generated Python
  (`WorkflowSource`) are *upstream drafts* and *renderings* of it, not rival
  sources of truth. Python task code is the escape hatch, not the primary model.
- Workflow availability flows into `WorkspaceContext.workflows` (§1.3).

### 3.2 Run binds Workflow to concreteness

- `Experiment.add_run(params=…)` / `add_runs(space)` create content-addressed
  Runs (`derive_run_id`). A `Run` binds: the referenced workflow, concrete
  `params`, environment (`profile`/`config_hash`), and time
  (`_ops/run.json` started/finished). This is the existing model — reuse as-is.
- Execution persists node-level outputs under `executions/<exec_id>/workflow.json`
  (resume seed) — unchanged.

### 3.3 Run → Artifact → Knowledge

- Run outputs become `ArtifactAsset`s via `ctx.artifact.save(...)`
  (`workspace/assets/accessors.py`) with `Producer(run_id, execution_id,
  task_id, inputs)` lineage — **this linkage already exists**; the gap is only
  that nothing consumes it for knowledge.
- Artifacts + logs become `KnowledgeItem`s via the **Artifact Interpreter** (§7.5)
  emitting source-linked items (§5). This is the missing edge the loop needs.
- Knowledge informs the next `ExperimentPlan` (Planner reads
  `WorkspaceContext.knowledge`) or a `WorkflowChangeProposal` (§8). Loop closes.

---

## 4. Artifact & provenance bridge

### 4.1 One artifact concept, two stores — bridge, don't fork

Audit finding: artifacts live in **two** systems — workspace `ArtifactAsset` +
`assets.json` (run outputs) and harness `ArtifactRef` + `harness.sqlite`
(agent-pipeline products). The bridge (P1) is a **mapping + unified read**, not a
merge-by-rewrite:

- Harness stages register their durable products into the workspace manifest via
  the workspace API (so a plan's `experiment_spec`, `workflow_ir`, `final_report`
  become queryable `ArtifactAsset`s with a `kind` mapping). `harness.sqlite`
  remains the per-run deep lineage; `scan_assets` becomes the single
  cross-cutting query.
- `content_hash` (`compute_content_hash`, "sha256:…") is the shared identity that
  lets a harness `ArtifactRef.sha256` and a workspace `ArtifactAsset.content_hash`
  refer to the same bytes — this is the join key for the bridge.

### 4.2 Per-type expected metadata

Every artifact carries the base `Asset` fields (id, scope, path, content_hash,
`Producer`, tags) plus a typed `metadata` block. Types map onto existing/known
`ArtifactKind`s:

| Type | `kind` | Expected metadata | Agent operation |
|---|---|---|---|
| file | `artifact`/`output_file` | mime, size, role | summarize, diff vs prior version |
| table | `dataset`/`table` | schema (cols+dtypes), rowcount, units | extract metrics, compare across runs |
| figure | `plot` | caption, axes, source dataset id | summarize, describe trend (vision) |
| model | `checkpoint`/`model` | architecture, params, training run_id, metric snapshot | record provenance, compare metrics |
| log | `log`/`stdout`/`stderr` | stream, exit_code, run/exec id | classify failure → FailureAnalysis |
| report | `final_report`/`experiment_report` | sections, linked artifact ids | extract Findings/Decisions |
| metric | `metric` | name, value (`ParameterValue`), tolerance, run_id | compare to acceptance criteria |
| dataset | `dataset` | format, n_records, content_hash, source | validate, summarize, derive Observation |

`ArtifactKind` (`harness/schemas/artifact.py:26`) is an **open `str` alias** (any
non-empty string), **not** a closed `Literal` — so new kinds (`table`, `metric`,
`model`, `change_proposal`) are just new strings: there is no vocabulary to
"extend" and no schema change to make. Reuse a well-known kind where one fits and
mint a new string where none does.

### 4.3 How agents use artifacts (source-linked, never paraphrased-as-truth)

- **Summarize / describe** → produce an `Observation` KnowledgeItem whose
  `sources` cite the artifact `content_hash` + `run_id`.
- **Compare** (across runs / versions) → produce a `Finding` citing both
  artifacts.
- **Validate** (against `acceptance_criteria` / `expected_metrics`) → a
  `test_result`-style artifact + (on mismatch) a `FailureAnalysis`.
- **Extract metric** → a `metric` artifact + `ParameterRationale` if the value
  feeds a decision.

The cite mechanism is the typed OKF out-edge (reuse `Folder.append_link` /
`out_edges`, `workspace/folder.py:727,299`), given an edge **role** (`derived_from`,
`cites`, `supersedes`). This is the single small primitive several slices depend
on (§9 P0).

---

## 5. Knowledge as long-term project context

### 5.1 Typed, source-linked — not generic notes

Today knowledge = untyped `Note`/`ReferenceConcept` linked by plain markdown
links, plus an ad-hoc `experiment-record-*` Note that encodes ids in its
*name/body text* (`server/plan_runtime/record.py:204`). This spec replaces the
ad-hoc bridge with a typed concept.

`KnowledgeItem` is a new OKF concept type (`@concept_type("knowledge.item")`,
registered in `knowledge/types.py`, stored as a `Folder` with
`meta.yaml`+`index.md` exactly like `Note`). It does **not** introduce a new
storage substrate — it is a `Note` with a typed head:

```python
KnowledgeKind = Literal[
    "Observation", "Decision", "Assumption", "Constraint", "Finding",
    "FailureAnalysis", "ProtocolNote", "ParameterRationale", "OpenQuestion",
]

class KnowledgeMeta(ConceptMeta):              # extends workspace/concept_meta.py
    kind: KnowledgeKind
    sources: list[SourceRef]                   # REQUIRED, non-empty (see 5.2)
    status: Literal["active", "stale", "superseded", "conflicting"] = "active"
    supersedes: list[str] = []                 # KnowledgeItem ids
    confidence: float | None = None
    created_by: str                            # user / agent:name
```

The body (`index.md`) holds the human-readable content; `meta.yaml` holds the
typed head. Reuse `Bundle` for traversal/index.

### 5.2 Source attribution (mandatory invariant)

`sources` is **required and non-empty** — a KnowledgeItem with no source fails
validation loudly (no silent empty). A `SourceRef` is a typed pointer to an
existing canonical object:

```python
class SourceRef(BaseModel):
    kind: Literal["artifact", "run", "experiment", "file", "decision", "agent_action", "reference"]
    ref: str            # content_hash | run_id | path | proposal_id | reference id
    span: str | None = None   # line range / cell / figure region, when applicable
```

Persisted as typed OKF out-edges so the knowledge graph is queryable both ways
(reuse `out_edges`). This is what makes every reasoning artifact *citeable* and
satisfies the harness-goal §17.6 provenance acceptance.

### 5.3 Who consumes knowledge

| Consumer | Uses knowledge for | Reads |
|---|---|---|
| Experiment planning (§7.2) | prior Findings/Decisions/Constraints to shape the next plan | `WorkspaceContext.knowledge` |
| Workflow recommendation (§7.3) | ProtocolNotes / past WorkflowChanges | knowledge + workflow validators |
| Run failure diagnosis (§7.4) | prior FailureAnalysis with matching signature | knowledge filtered by error class |
| Artifact interpretation (§7.5) | ParameterRationale / Assumptions to read a result correctly | knowledge for the run's lineage |
| Agent response generation (§6) | grounded, cited answers | knowledge as retrieval context |
| Workspace summaries (§7.1) | open questions, recent findings | `WorkspaceContext.knowledge` + `open_questions` |

### 5.4 Stale / duplicate / conflict detection

The **Knowledge Curator** (§7.6) computes, never silently mutates:

- **stale** — a `source` content-hash superseded by a newer artifact, or a `run`
  source now `failed`/deleted → flag `status="stale"`.
- **duplicate** — high text + same-`kind` + overlapping-`sources` similarity →
  propose merge.
- **conflict** — two `active` items of the same `kind` with contradictory claims
  over the same sources → emit `knowledge.conflict.detected` → a ChangeProposal to
  supersede/merge (§8). Conflicts are **never** auto-resolved; they become
  proposals that preserve both source attributions.

---

## 6. Agent operation model

### 6.1 Observe → Interpret → Propose → Act → Record

| Stage | Inputs | Outputs | Allowed ops | Forbidden | Required source links | Approval |
|---|---|---|---|---|---|---|
| **Observe** | focus + `WorkspaceContext` (§1) | structured observation set | read context, `scan_assets`, read knowledge, read logs | any write; reading outside workspace root | — | none |
| **Interpret** | observations + knowledge | analysis (typed) | LLM reasoning, metric extraction | asserting inferred values as user facts | cite every artifact/knowledge used | none |
| **Propose** | analysis | `ChangeProposal` (§8) or advisory text | emit proposal artifact | mutating canonical state | proposal must link evidence + knowledge | n/a (proposal *is* the gate input) |
| **Act** | approved proposal | result artifacts + events | run executor / lifecycle / curation op | any op outside the proposal's `affected_objects`; unapproved high-risk op | action cites the proposal id | per §8 level |
| **Record** | action result | artifacts, `WorkspaceEvent`s, KnowledgeItems | write artifacts, append events, write knowledge | overwriting prior artifacts (new version only) | every record links its producing action | none |

These map onto existing machinery: Observe = `workspace.context` assembler;
Interpret/Propose = `AgentRunner` loops (`agent/runner.py`) or harness mode
stages; Act = `harness/executors/` + run lifecycle + `workspace/curation/`;
Record = workspace assets + `WorkspaceEvent` log + KnowledgeItems.

### 6.2 Modes (escalating authority)

| Mode | Existing basis | Can write? | Gate |
|---|---|---|---|
| **read-only analysis** | `ChatLoop` / `InteractiveLoop` with `readonly_tools` (`agent/loops/interactive/tools.py`) — *already exists* | no | none |
| **advisory proposal** | agent loop emitting a `ChangeProposal`, no executor | proposal artifact only | human reads, may discard |
| **guarded execution** | `ChangeProposal` → `ApprovalGate` (`harness/stages/approval_gate.py`) → executor | yes, post-approval | explicit approval |
| **autonomous low-risk maintenance** | policy-gated auto-approver (`harness/policy/`, `auto_grant_approver`) restricted to a low-risk op allow-list | yes, bounded | policy auto-approves, still recorded |

The single rule that separates modes: **a high-risk operation (§8.1) MUST become
a `ChangeProposal` before execution**, regardless of mode. Autonomous mode only
auto-*approves* low-risk proposals; it never skips the proposal+record trail.

---

## 7. AI-assisted features (first useful set)

Each feature is a **consumer of `WorkspaceContext` + a producer of
artifacts/proposals/knowledge** — none owns canonical state. Mapping to existing
infra is explicit so these are seams, not new stacks.

| # | Feature | Reuses | Adds | Output |
|---|---|---|---|---|
| 7.1 | **Workspace Copilot** | `WorkspaceContext` (§1), `InteractiveLoop`, health flags (§1.4) | a read-only summary mode | workspace summary + ranked next-actions (advisory) |
| 7.2 | **Experiment Planner** | **`PlanMode` already implements this** (`harness/modes/plan.py`) | feed it `WorkspaceContext.knowledge`; emit plan as `experiment.plan.updated` | experiment_spec + workflow IR + tests (existing) |
| 7.3 | **Workflow Builder / Repair** | WorkflowIR validators (`harness/validators/`), capability registry, curation workflow-half (`workspace/curation/`) | a repair stage that diffs IR vs validation findings | `WorkflowChangeProposal` (§8) |
| 7.4 | **Run Monitor** | `read_ops` (`run.py:225`), `scan_assets`, log artifacts | failure classifier over `run.failed` events | run summary + `FailureAnalysis` knowledge + retry/resume/fix proposal |
| 7.5 | **Artifact Interpreter** | `scan_assets`, per-type metadata (§4.2), vision via router | metric/summary extractor | source-linked `Observation`/`Finding`/`metric` |
| 7.6 | **Knowledge Curator** | Bundle, KnowledgeItem (§5), similarity | stale/dup/conflict detector (§5.4) | `knowledge.conflict.detected` + merge/supersede proposals |

Note 7.2: the Experiment Planner is **not new** — `PlanMode` is the running
implementation. The integration work is to *feed it WorkspaceContext* (so it
plans grounded in prior knowledge and existing runs) and to *route its output
through the unified event + record path* it already partly uses via
`materialize_plan_records`.

---

## 8. ChangeProposal protocol

### 8.1 When an action MUST become a ChangeProposal (high-risk set)

- modifying a **Workflow** (IR or generated source)
- modifying an **Experiment parameter space**
- deleting or replacing **Artifacts**
- changing **canonical Knowledge** (supersede/merge/delete)
- **rerunning or canceling Runs**
- **moving or rewriting workspace assets** (the destructive `workspace/curation/`
  ops — `reorg.delete_folder` / `move_run` / `rehome_asset`, `reorg.py:92/38/65`,
  which are **currently ungated and have no production caller**; wiring them behind
  this gate is a **P2.1** deliverable, *not* an existing fact — the only approval
  gate today is harness-side (`BoundWorkflow` side-effects), unreachable from
  workspace)
- changing **public APIs or schemas**

Low-risk ops (read, summarize, create a *new* draft KnowledgeItem with sources,
create a *new* experiment plan draft) do not require a proposal but are still
recorded.

### 8.2 Schema

`ChangeProposal` is **both** a harness artifact kind (the open-`str`
`change_proposal`, so it lives in the lineage spine) **and** a durable record under
the owning `AgentTask` (so it survives review out-of-band). *Ownership is crisp:*
the **schema lives in `harness/schemas/`**; the **`AgentTask` is a server-layer
surface** (`server/routes/agent_tasks.py` `PersistedAgentTask`), so the **server**
orchestrates persisting the proposal under its task — `workspace` and `agent` never
reference `ChangeProposal` (both sit below/beside harness in the DAG). It reuses
`ApprovalRequest`/`ApprovalDecision` (`harness/schemas/approval.py:41,54`) for the
gate — the proposal is the *payload* the gate decides on.

```python
class ChangeProposal(BaseModel):
    id: str
    intent: str                                  # what & why, structured
    current_state: StateSnapshot                 # refs to affected objects' current versions
    proposed_change: ChangeSpec                  # typed diff (workflow IR diff / param-space delta / knowledge merge / run op)
    affected_objects: list[ObjectRef]            # the ONLY objects Act may touch
    expected_benefit: str
    risks: list[str]
    reversibility: Literal["reversible", "partially", "irreversible"]
    approval_level: Literal["auto", "user", "elevated"]   # maps to §6.2 modes / policy
    evidence: list[ArtifactRef]                   # cited artifacts (required for execution proposals)
    knowledge: list[SourceRef]                    # cited KnowledgeItems
    execution_result: ProposalOutcome | None = None   # filled AFTER approval+act
```

`approval_level` is derived from a `policy` (`harness/policy/`): the high-risk set
(§8.1) defaults to `user`/`elevated`; the low-risk maintenance allow-list maps to
`auto`. `reversibility="irreversible"` forces at least `user`.

### 8.3 Lifecycle (reuses existing gate + events)

```text
agent.proposal.created (event, P✓)
  → ApprovalGate decides (approval_requested → granted|rejected, harness.sqlite)
      rejected → recorded, no Act; proposal kept for audit
      granted  → Act (executor / lifecycle / curation op, bounded to affected_objects)
                 → execution_result filled
                 → agent.action.completed (event, P✓)
                 → resulting Run/Artifact/Knowledge recorded with proposal_id back-link
```

Every step is append-only and traceable; a rejected proposal is **not** deleted —
it is the record that the agent proposed and a human declined.

---

## 9. Implementation roadmap

Small cooperation-focused slices. No broad rewrite. Each builds on the prior.

### P0 — make the loop expressible (state, events, linkage, skeleton)

| Slice | Goal | Reuse | New | Files (likely) | Tests | Risk | Exit criteria |
|---|---|---|---|---|---|---|---|
| **P0.1 Typed provenance edge** | give OKF out-edges a `role` + fix Bundle nested-mount | `folder.py` links/out_edges/append_link, `bundle.py` | `EdgeRole` enum | `workspace/folder.py`, `bundle.py` | edge round-trip; nested mount under Run | low | a concept can typed-link to a Run/Artifact and resolve back |
| **P0.2 WorkspaceContext assembler** | one canonical read-model (§1) | `read_ops`, `scan_assets`, `Bundle`, folder tree | `workspace/context.py`, `WorkspaceContext` schema | `workspace/context.py`, `server/routes/` (`GET /context`), `cli` (`molexp context`) | assembler unit; route parity with old `/info` | low | server + CLI + UI read identical context |
| **P0.3 WorkspaceEvent spine** | append-only cross-object log (§2) | **Layer-0 SQLite append-log primitive extracted from `harness/store/_sqlite.py`**, cited by both harness + workspace (§2.1 OPEN DECISION — do NOT copy the store) | `workspace/events.py` (`WorkspaceEventLog`) + the extracted Layer-0 primitive | `workspace/events.py`, emit sites in `run_lifecycle.py`, `assets/accessors.py`, `materialize.py` | append/seq/ordering; emit on run+asset lifecycle; single store impl (no duplicate) | medium-high | `run.*` / `asset.added` / `knowledge.created` queryable per workspace, **zero store duplication** |
| **P0.4 Run→Artifact→Knowledge link + KnowledgeItem** | typed `knowledge.item` concept with required sources (§5) | `concepts.py`, `concept_meta.py`, `knowledge/types.py`, P0.1 | `KnowledgeMeta`, `SourceRef` | `workspace/concepts.py`, `knowledge/types.py`, `server/routes/knowledge.py` | source-required validation; provenance query | medium | a KnowledgeItem cites a run/artifact and is reachable from it |
| **P0.5 AgentIntent / ChangeProposal skeleton** | schemas + gate wiring, no executors yet (§6,§8) | `ApprovalGate`, `ApprovalRequest/Decision`, `ArtifactStore` | `ChangeProposal`, `ChangeSpec`, `ProposalOutcome` | `harness/schemas/change_proposal.py`, `harness/stages/` | proposal→gate→reject/grant recorded | medium | a proposal can be created, gated, and audited end-to-end (dry) |
| **P0.6 Workspace summary assistant** | Copilot read-only over context (§7.1) | `WorkspaceContext`, `InteractiveLoop`, health flags | a summary mode | `harness/modes/` or `agent` feature, `server` route, UI panel | summary shape; next-action ranking | low | UI shows a grounded workspace summary + next actions |

### P1 — the assistive features

| Slice | Goal | Reuse | New | Files | Tests | Risk | Exit criteria |
|---|---|---|---|---|---|---|---|
| **P1.1 Experiment Planner ← context** | feed `WorkspaceContext.knowledge` into PlanMode; emit `experiment.plan.updated` | `PlanMode`, `materialize_plan_records`, P0.2/P0.3 | context→plan input adapter | `harness/modes/plan.py`, `server/plan_runtime/` | planner consumes prior Findings | medium | a plan references existing knowledge & runs |
| **P1.2 Workflow validator / repair proposal** | diff IR vs validation, propose patch (§7.3) | WorkflowIR validators, capability registry, P0.5 | repair stage | `harness/stages/`, `harness/validators/` | broken-IR → repair proposal | medium | invalid workflow yields a reviewable `WorkflowChangeProposal` |
| **P1.3 Run failure analyzer** | classify `run.failed`, emit FailureAnalysis + retry/resume/fix proposal (§7.4) | `read_ops`, log assets, P0.3/P0.4/P0.5 | failure classifier | `harness`/`server` monitor feature | failure-class → knowledge + proposal | medium | a failed run produces a cited FailureAnalysis |
| **P1.4 Artifact summary + metric extraction** | per-type interpreter → source-linked knowledge (§7.5) | `scan_assets`, §4.2 metadata, router vision | interpreter stages | `harness`/`server`, `workspace/assets` | metric extraction; source links present | medium | an artifact yields an `Observation`/`metric` citing it |
| **P1.5 Knowledge stale/conflict detection** | curator detectors (§5.4) | Bundle, KnowledgeItem, P0.4 | similarity + flags | `workspace`/`harness` curator | stale/dup/conflict cases | medium | conflicts surface as `knowledge.conflict.detected` |

### P2 — execution & learning

| Slice | Goal | Reuse | New | Risk | Exit criteria |
|---|---|---|---|---|---|
| **P2.1 Guarded execution** | proposal→approval→executor→record for high-risk ops (§6.2,§8) | `ApprovalGate`, `executors/`, lifecycle, curation | policy→approval_level binding | high | an approved workflow/param change executes & records |
| **P2.2 Autonomous low-risk maintenance** | policy auto-approve a bounded allow-list (§6.2) | `harness/policy/`, `auto_grant_approver` | low-risk op registry | high | stale-flag cleanup runs autonomously, fully recorded |
| **P2.3 Multi-agent planning/review** | adversarial plan + review panel | `AgentGateway`, `plan_agents` | reviewer roles | medium | a plan is independently reviewed before approval |
| **P2.4 Project-level learning** | mine repeated runs → ProtocolNotes/ParameterRationale | knowledge, runs, artifacts | aggregator | medium | recurring outcomes become reusable knowledge |

---

## 10. Coordination invariants (enforceable contracts)

These restate the task rules as checks the design must keep true (candidates for
import-guard / unit tests, then promotion to `CLAUDE.md`):

1. **No parallel architecture** — every new module is a seam over an existing
   subsystem (§0). A second copy of an existing store/registry/mode is a defect.
2. **Canonical state is workspace-on-disk** — agent memory and frontend state are
   never read as truth (§0, §1.1).
3. **Agent never directly mutates high-risk state** — the §8.1 set requires a
   `ChangeProposal` + gate before any executor (§6.2, §8).
4. **Every KnowledgeItem has ≥1 SourceRef** — empty sources fail loudly (§5.2).
5. **Every Run output is connectable to an Artifact** — via `Producer` lineage,
   already true; the bridge keeps it true cross-store (§3.3, §4.1).
6. **Every Artifact used in reasoning is citeable** — by `content_hash`/`run_id`
   source ref (§4.3, §5.2).
7. **Every agent action is traceable** — `agent.intent.received` →
   `agent.proposal.created` → `agent.action.completed` chain with provenance
   (§2.2, §8.3).
8. **Every ChangeProposal is reviewable before execution** — and a rejected
   proposal is preserved, not deleted (§8.3).
9. **Structured objects over free-form chat** — features emit
   proposals/artifacts/knowledge, not prose-as-state (§7).
10. **No silent invalid state** — stale/missing/conflict is *flagged*, never
    hidden by fallback (§1.4, §5.4); this extends the project's existing
    "fail loudly / no fallback" rule.

---

## Relationship to the other notes

- `harness-goal.md` — owns the harness internals (ArtifactRef, HarnessEvent,
  WorkflowIR/BoundWorkflow split, executors, validators, audit). This spec
  **consumes** those and adds the cross-layer coordination + agent operation
  model. Where they overlap (events, provenance, proposals/approval), this spec
  defers to harness-goal for the *intra-run* mechanism and specifies the
  *inter-object* coordination on top.
- `architecture.md` — current-state map; the audit that motivated this spec.
- New invariants here graduate to `CLAUDE.md` via `/mol:note` **only after a
  slice lands and proves them** — not before.
