---
mol_project:
  name: molexp
  language: mixed
  build:
    install: 'pip install -e ".[dev]"'
    check: "ruff format --check src/ tests/ && ruff check src/ tests/ && ty check src/"
    # Mirrors ci.yml `test` + pre-push pytest hook (quiet + coverage xml).
    test: "pytest tests/ -q --cov=src/molexp --cov-report=xml"
    test_single: "pytest {path} -v"
    coverage: "pytest tests/ --cov=src/molexp --cov-report=term-missing"
  arch:
    style: layered
    rules_section: "## Architecture"
  doc:
    style: google
  science:
    required: false
  stage: experimental
  ci:
    config: .github/workflows/ci.yml
  dev:
    command: "cd ui && npm run dev:mock"
    url: "http://localhost:5173"
    ready_pattern: "Local:"
    url_pattern: "Local:\\s+(https?://\\S+)"
    ready_timeout: 90
  notes_path: .claude/notes/notes.md
  specs_path: .claude/specs/
---

# CLAUDE.md

molexp is an agent-assisted scientific-workflow platform for FAIR research — Python (FastAPI + PydanticAI + a self-owned workflow engine) + TypeScript/React 19 UI, shipped as one wheel.

## Where things live

- Python source: `src/molexp/` · TS source: `ui/src/`
- Tests mirror source: `tests/` ↔ `src/molexp/`, `ui/src/` for the UI
- Public docs: `docs/` · in-flight specs: `.claude/specs/` (gitignored)
- **Detailed module map: `.claude/notes/architecture.md`** — keep that file as the index; CLAUDE.md only carries invariants

## Hard invariants (do not change casually)

- **MolRec layering:** a molexp **Run is a host**, not a MolRec record (`run.json` / `_ops/run.json` ≠ molrec `meta` / `status`). Scientific packages follow the **external molrec spec** (Zarr root + optional `metrics/metrics.jsonl` buffer) — molexp does not ship a molrec module or re-host the contract. `metrics/index.json` under a run is a **host series cache only**. Charts: molplot only (JSONL buffer).
- **Layer DAG** (enforced by `tests/test_<layer>/test_import_guard.py`): `services → harness + agent + workflow + workspace`, `harness → agent + workflow + workspace`, `agent + workflow → workspace`. Two **sibling** layers (agent, workflow) above workspace; harness sits **above** all three; **`molexp.services`** (application services shared by CLI and server: `plan_runtime` / `curate_runtime` / `operator_config` / `agent_task_store` / `agent_context` / `approval_notify`) sits above harness and below CLI/server — CLI and server import services, never each other. The single sanctioned `harness → agent` import target is `molexp.agent.router` (the SDK-free Protocol module). Any other arrow is an architectural defect.
- **Workflow public API**: decorator and OOP styles on the same `WorkflowCompiler`; task bodies may be plain `def` **or** `async def` (sync bodies run via `asyncio.to_thread`, never blocking same-level parallelism). `WorkflowCompiler` / `TaskContext` / `WorkflowRuntime` / `execute_run` / `aexecute_run` are re-exported lazily at the top level (`molexp.WorkflowCompiler`, …) — `import molexp` must stay light (no `pydantic_graph` in `sys.modules`; the dependency itself was removed).
- **Interface naming contracts (frozen — reject new spellings).** `params` is the one kwarg spelling across the hierarchy (`add_experiment(params=…)` / `add_run(params=…)` / `run(wf, params=…)`; `parameters=` survives only as a DeprecationWarning alias on `add_run`). `cancel` is the canonical stop verb (see verb law); the operator config key is `agent.model` (`~/.molexp/config.json`, bridged into `molexp.config` by `services/operator_config.py` — CLI and server share one loader). `write_reference_meta` is the canonical Concept-meta writer spelling (`write_ref_meta` survives only as a DeprecationWarning alias).
- **`Workspace → Project → Experiment → Run` is a `Folder` family.** Generic CRUD (`add_folder / get_folder / has_folder / list_folders / remove_folder`) + typed semantic sugar per subclass (`add_project / add_experiment / add_run / …`). Index filenames auto-derived as `cls.__name__` snake_case + `.json`. `add_*` is idempotent on slugified name.
- **Agent public surface**: `AgentRunner`, `AgentLoop`, `AgentRunResult`, `AgentRuntime`, `AgentSession`. Loops are plain `async def run(*, runtime, sink, user_input) -> None`; events flow through the injected `AsyncIteratorEventSink`. Two loops ship — `ChatLoop` (one round-trip) and `InteractiveLoop` (emergent tool loop); pipeline orchestration moved to harness. ("Loop" is the agent-layer LLM-conversation concept; "Mode" is reserved for `molexp.harness.Mode` orchestration.)
- **OpenAPI surface** under `src/molexp/server/routes/`; the UI regenerates against it (`npm run generate:api`). Don't hand-edit `ui/src/api/generated/`.
- **Private subpackages** — never import from outside their owning layer: `workflow/_engine/` (the self-owned execution engine), `agent/_pydanticai/`, `agent/mcp/`.

## Architecture

```
       harness          (experiment orchestrator — artifact lineage, audit,
         │                stages, executors, approval gates)
         │
         ├─ uses ─→ agent     (pydantic-ai facade + LLM-only loops)
         ├─ uses ─→ workflow  (graph engine, content-addressed cache)
         └─ uses ─→ workspace (pure storage)

     agent ─→ workspace        (sessions on disk)
     workflow ─→ workspace
```

Server + CLI sit on top of **`molexp.services`** (the application-service layer: `plan_runtime` / `curate_runtime` / `operator_config` / `agent_task_store` / `agent_context` / `approval_notify` — one backend code path for "Python 操作 = UI 操作"), which sits on top of harness/agent/workflow/workspace; UI is downstream of the server's OpenAPI. Cross-layer primitives (`molexp.path`, `molexp.profile`, `molexp.entry`, `molexp.atomicio`, `molexp.ids`, …) sit above workspace and may be cited from any layer. **`molexp.knowledge`** is a **bottom** layer that owns only the Open Knowledge Format concept-type registry (`@concept_type` / `register_concept_type` / `resolve_concept_type`); the OKF-native storage substrate lives in `molexp.workspace`, which uses this registry to reconstruct typed `Folder` subclasses from each Concept's persisted `type`. `knowledge` is a peer of `workspace` and imports neither it nor any upstream layer. `molexp.config` is the process-global in-code config — a live `molcfg.Config` instance defined in `molexp/__init__.py` (LLM keys etc., registered in code, never from env); `molexp.profile` is the separate file-based, per-run profile config (`ProfileConfig` / `MolCfg` / `load_molcfg`).

### Layer charters

**`molexp.knowledge`** — bottom layer; the open **concept-type registry** (single responsibility). Not a storage substrate — the OKF-native storage lives in `molexp.workspace`.
- Owns: only the generic, open concept-type registry (`@concept_type` / `register_concept_type` / `resolve_concept_type`) used by `workspace` to reconstruct typed `Folder` subclasses from each Concept's `meta.yaml` `type`. Upstream layers register their own Concept types here without `knowledge` importing them; an unknown type resolves to a caller-supplied default. The package is just `types.py` + a slim `__init__.py`.
- **Allowed imports**: stdlib / pydantic only (no pyyaml needed). **MUST NOT** import `workspace` or any upstream layer (`workflow` / `agent` / `harness` / `services` / `server` / `cli` / `plugins`) — enforced by `tests/test_knowledge/test_import_guard.py`, an **AST source scan** (not a `sys.modules` probe: `molexp/__init__.py` eagerly loads `workspace`, so a runtime probe is unsatisfiable).

**`molexp.workspace`** — bottom; pure storage.
- Owns: `Folder` base + the `Workspace/Project/Experiment/Run` subclasses, typed exceptions (`*NotFoundError` / `*ExistsError`), atomic JSON I/O (`atomic_write_json`), the `Asset` family + per-scope `AssetManifest` (`assets.json`) + the `assets.scan` manifest-scanning query layer (`scan_assets` / `get_asset` / `find_by_content_hash`), `Params` / `ParamSpace` / `GridSpace` / `UniformSpace`, the **unified Target family** (`ComputeTarget` is the persisted base; `LocalTarget` / `RemoteTarget` are its address-view subclasses; `resolve_compute_target` is the single named-target resolution path for CLI and server; `SSHSession` for transport caching), `RunContext`, `RunSet` / `RunSetResult` (sweep container: `experiment.sweep(...)` → `runset.execute()` → `to_records()` / `min_by()`; execution delegates through the `set_run_executor` inversion seam so workspace never imports workflow), zombie reaping (`run_reaper.reap_zombie_run`, called by every CLI **and** server verb entry), run-lifecycle verb cores (`lifecycle_ops.cancel_run`; two-phase `prune.plan_execution_prune` / `apply_execution_prune`), execution→knowledge (`harvest.harvest_run`), the workspace event log (write **and** read: `read_workspace_events`), the OKF Concept surface (`Note` / `ReferenceConcept` + `ReferenceMeta` + `write_reference_meta`, the `Bundle` façade, `ZoteroItem` / `read_zotero_items` — surfaced as `molexp knowledge import-zotero`), and one singleton folder accessed as a lowercase property: `ws.cache` (`CacheFolder` → `as_cache_store()` adapter). **There is no derived SQLite asset index** — asset queries scan the authoritative per-scope `assets.json` manifests (+ directly-registered `assets/<id>/asset.json` records); the former `AssetCatalog` / `catalog/index.sqlite` was removed (`workspace-git-projection-01`).
- **Notes + literature are OKF Concepts (`Note` / `ReferenceConcept`), reached via the `Bundle` façade** — directories whose path is their identity (bib fields in `meta.yaml` via `ReferenceMeta`; PDFs *pointed at*, never copied). The legacy per-scope `Library` (record-`Reference` / `ReferenceStore` / `NoteAsset` / `LibraryIndex` / `.library` properties / `search_library` agent tool / `/api/library` routes / UI Library page) was removed in wsokf-11; greenfield, no migration.
- MUST NOT: import any upstream `molexp` layer (`workflow` / `agent` / `plugins` / `services` / `server` / `cli`). Allowed `molexp.*` imports are only `_typing` / `profile` / `path` and cross-layer primitives (`mollog`, `molcfg`). MUST NOT define workflow- or agent-shaped types (no `WorkflowSnapshotRef`, no `Agent` / `AgentSession` / `PlanFolder`). MUST NOT write to disk in `__init__` — all I/O is lazy.
- `import molexp.workspace` must never pull `molexp.workflow`, `molexp.agent`, `pydantic_ai`, or `pydantic_graph` into `sys.modules`.

*Identity & persistence law (workspace).*
- **Three orthogonal id layers, no fourth.** Uniqueness → `generate_id()` (UUID[:8]) / asset UUIDs. Reproducibility & dedup → content hashes: `config_hash` is a run's *one* config identity, `compute_content_hash()` (`"sha256:…"`) addresses artifacts. Location → `AssetScope(kind+ids)` + `execution_id` (`exec-{run_id}[-N]`). A run carries exactly one identity — never add a second parallel run-fingerprint type alongside `config_hash`.
- **Run vs Execution.** `Run` = reproducible logical unit (params + workflow); `Execution` = one physical attempt, identified `exec-{run_id}[-N]`, N-per-run. Per-attempt state lives under `executions/<exec_id>/` and is self-describing (it duplicates the run's `execution_history` summary on purpose); cross-attempt state stays at run level. An Execution persists completed-task outputs at **workflow-node granularity**, so it can be resumed in place (next bullet).
- **Two ways to (re-)execute a Run, both on the same `run_id`; there is NO new-Run operation.** (1) **resume** — reopen the *existing* `exec_id`: seed the already-completed task outputs from that execution's persisted node-level state, recompute only the unfinished/failed nodes and their downstream. Same `exec_id`; its `ExecutionRecord` flips back to `running` and `finished_at` clears. (2) **rerun** — open a *new* `exec-{run_id}-N`: fresh `ExecutionRecord`, clean attempt from the top of the graph (content-addressed cache may opportunistically hit, but seeding is not the semantic; `--rerun --fresh` / `rerun=True, fresh=True` additionally bypasses the cache *read* for that attempt — results are still written back). Cloning params into a fresh `run_id` is NOT a molexp operation. Only two verbs exist — `resume` / `rerun` — and they stay distinct everywhere (CLI flag, server route, code). **Success requires evidence**: a reopened attempt that runs nothing flips nothing — the exit path is three-state (engine signalled FAILED → failed; engine signalled succeeded → succeeded; no signal on a previously failed/cancelled run → status preserved, the `ExecutionRecord` closes as `aborted`), and a SUCCEEDED terminal state clears `metadata.error`. Task failure always writes `executions/<exec_id>/error.txt`.
- **Run status × verb selection (canonical — three orthogonal verbs).** A `Run` is `pending → running → succeeded | failed | cancelled`; a stale `running` is reaped to `failed` *before* any verb decides — same-host with a dead owner PID, or **cross-host only when the heartbeat label is stale** (`labels.heartbeat`, refreshed every 30 s by the run lifecycle; stale ≥ 10 min — a cross-host run with a fresh or absent heartbeat is never reaped, that's a live HPC job). The retryable domain (`failed`/`cancelled`) has one source of truth: `workspace.run.RETRYABLE_STATUSES` / `Run.is_retryable`, consumed by both CLI and server. Each verb owns a **disjoint** job: **run** (no verb) → start what has not run (create missing + run `pending`); **resume** → `failed`/`cancelled` only, reopen the last execution + seed completed nodes (continue from where it stopped); **rerun** → `failed`/`cancelled` only, fresh `exec-{run_id}-N` (re-execute from the top). Everything outside a verb's domain is **skipped**: `run` leaves `failed`/`cancelled`/`succeeded`/`running` alone (retrying is an explicit verb, never implicit); `resume`/`rerun` skip `pending` (run's job), `succeeded` (done) and a *live* `running` run (one Run = one ownership stamp + one status → never a second concurrent execution; **cancel** it first to intervene — `cancel` is the canonical verb everywhere: CLI `molexp runs cancel`, server `POST /{run_id}/cancel`; `POST /{run_id}/kill` survives only as a deprecated alias). CLI (`molexp run` / `--resume` / `--rerun [--fresh]`, profile-gated), server (`POST /{run_id}/resume|rerun[?fresh=true]`, 409 outside the failed/cancelled domain) and the Python one-step API (`molexp.execute_run(wf, run, rerun=…, fresh=…)` / `Run.execute` / `RunSet.execute`) follow the same rule via the same shared components — reaping included (`workspace.run_reaper`). The CLI noun group stays plural (`molexp runs …`) deliberately: typer's flat per-level namespace means a `run` group would shadow the top-level `molexp run` execution verb.
- **One source of truth.** Entity `*.json` + per-scope `assets.json` asset manifests are authoritative. Every index (the children-index `*.json`, any future derived view) is *derived* — rebuilt by scanning the authoritative files, never the only copy, never consulted as truth. Asset queries scan the manifests directly (`assets.scan`); there is no derived asset index. Don't add per-container index files that re-encode what the authoritative manifests already hold.
- **No speculative code in `src/`.** A subsystem with zero production callers does not ship. "Future" capability with no live call-path belongs in docs or a branch, not the tree.

**`molexp.workflow`** — middle; graph execution engine.
- Owns: `WorkflowCompiler` (decorator + OOP, mutable) → `.compile()` → `CompiledWorkflow` (frozen, content-hashed). `Task` / `Actor` convenience bases; user-facing protocols are `Runnable` / `Streamable`; task bodies may be sync or async. `TaskContext` (the single context for batch + streaming bodies), `TaskTypeRegistry`, `TaskSnapshot` (AST-normalized hash), `Caching` (LRU), `WorkflowSnapshotRef`, `WorkflowResult` / `WorkflowExecution` (result-level status vocabulary is `succeeded`/`failed`; legacy persisted `"completed"` is normalized on read), `End` (molexp-owned sentinel in `workflow/types.py` — generic frozen dataclass, `End()` / `End(data)`; defined exactly once), `Next` (routing return value for `wf.branch` / `wf.loop`), `promote_callable()`, and `execute_run` / `aexecute_run` (one-step tracked execution + the `RunFailedError` / `RunNotExecutableError` pair; registers the workspace `set_run_executor` seam at import so `Run.execute` / `RunSet.execute` reuse the exact CLI execution path). Private `_engine/` is the self-owned execution engine — **pydantic_graph is no longer a dependency and must not be imported anywhere in `src/`** (enforced by `tests/test_workflow/test_engine_boundary.py`, a full-src AST scan).
- Uses workspace: `Workflow.execute(run=…)` takes a `workspace.Run`; `Caching` is backed by `ws.cache.as_cache_store()`; run-state JSON writes go through `workspace.atomic_write_json`.
- **resume is task-granular, caller-driven, and self-validating.** A failed run preserves completed outputs in `WorkflowResult.outputs`; the caller (CLI/server) re-seeds them via `runtime.execute(..., seed_outputs=…)`, sourced from the *existing* execution's persisted node-level state. Node state is persisted **incrementally** to `executions/<exec_id>/workflow.json` (per-task status + outputs + `snapshot_key` + `outputs_lossy` flag — not per-frame engine checkpoints; within-task checkpointing belongs to `ctx.workdir`). At seed time the engine drops (warn, recompute — never error) any seed whose recomputed `TaskSnapshot.key` differs (code changed), whose outputs were lossy-truncated, or which cannot be verified (pre-flag documents). Seeded nodes are recorded in `WorkflowState.seeded` and skip their body (`from_seed`); unknown seed names fail-fast. `rerun` simply calls `execute` with no seed (new `exec_id`). SubWorkflow inner runs execute with persistence off — `executions/<exec_id>/workflow.json` always describes the **outer** graph only.
- MUST NOT: import `agent` / `plugins` / `server` / `cli` / `services`. MUST NOT import `pydantic_graph` anywhere (the dependency is removed). No second `End` sentinel (single definition in `types.py`), no per-task node-class codegen. `Next` is public (the `wf.branch` / `wf.loop` routing return value, in `molexp.workflow.__all__`) but stays out of the frozen top-level `molexp.__all__` / lazy re-exports.

**`molexp.agent`** — sibling of workflow above workspace; pydantic-ai facade + LLM-only loops.
- Owns: the public surface (`AgentRunner`, `AgentLoop`, `AgentRunResult`, `AgentRuntime`, `AgentSession`). `AgentRuntime` is the frozen-dataclass bundle (`session` + `router` + `execution_env`) a loop receives at run time. Flat module list — `loop.py` / `runtime.py` / `session.py` / `session_storage.py` / `session_entry.py` / `events.py` (`AgentEvent` discriminated union + `AsyncIteratorEventSink`) / `compaction.py` / `execution_env.py` / `router.py` (Protocol) / `runner.py` / `folders.py`. Two concrete loops under `loops/`: `ChatLoop` (one round-trip) and `InteractiveLoop` (emergent tool loop driving `Router.stream_agentic`). Private `_pydanticai/` is the **sole** sanctioned `import pydantic_ai` site (`router.py`, `mcp.py`, `messages_codec.py`, …).
- Uses workspace only (`Folder`, `Workspace`, `Run`, `RunContext`, …) for on-disk session storage. **MUST NOT** import `molexp.workflow` or `molexp.harness` — those layers are siblings (workflow) and one layer above (harness); pipeline orchestration that used to live in agent moved to harness, reached through the `AgentGateway` Protocol.
- MUST NOT: import `molexp.workflow`, `molexp.harness`, `plugins`, `services`, `server`, `cli`. MUST NOT import `pydantic_ai` outside `_pydanticai/`. MUST NOT import `pydantic_graph` anywhere under `agent/`.
- Don't reinvent what pydantic-ai already does. Plain tools → `Agent(tools=[…])`; MCP servers → `Agent(toolsets=[MCPServerStdio(…)])`; retries → `Agent(retries=N)`; message history + structured output → pydantic-ai native. The agent layer (Session persistence, AgentEvent stream, on-disk folders) is what molexp owns; if pydantic-ai *cannot* do something, say so in the new module's docstring.
- `import molexp.agent` must not pull `pydantic_ai` or `pydantic_graph` into `sys.modules` until `AgentRunner.run()` is actually called.

**`molexp.harness`** — top; experiment orchestrator.
- Owns: artifact lineage + audit + execution machinery (`HarnessRunContext`, `Stage`, `StageRunner`, `ArtifactStore` + `FileArtifactStore`, `EventLog` + `SQLiteEventLog`, `ArtifactLineageStore` + `SQLiteArtifactLineageStore`, **`ApprovalStore` + `SQLiteApprovalStore`**, `CapabilityRegistry`, executors, validators, `ApprovalGate`, `generate_audit_report` + replay, **`workflow_recovery`**, curation + lifecycle capability catalogs). Run-level provenance (params, config hash, code/script identity) is owned by **workspace** (`RunMetadata` + per-scope `AssetManifest`); harness lineage covers agent-pipeline artifacts only and stamps each edge with stage + workspace `run_id`. The plan pipeline is **two-phase planning→realization**, composed by `PlanOrchestrator` (`harness/modes/plan_orchestrator.py`); stages live under `harness/stages/`. **Phase 1 — interactive planning**: the orchestrator drives the agent-layer Pi loop (`agent.loops.InteractiveLoop`) with the harness plan-tool surface (`harness.plan_tools`: `as_loop_tool`-adapted task-board tools + a single-point side-effect gate) and an `PlanFormValidator` `should_stop` guard — a malformed task board is **never surfaced to a human** (the guard denies termination and steers the violations back to the planning agent). After the loop, `PlanReachabilityProbe` read-only-annotates each board task's feasibility (molmcp grounding, no binding), a deterministic form guard re-checks, the current `ExperimentPlan` (spec + board) is persisted as the `experiment_plan` review subject, and a **hard** `StepAuditLoop` review gate runs store-first (suspends with `ApprovalPendingError` + a durable pending record; a stored grant replays). On grant, `freeze_experiment_plan` writes the content-addressed `frozen_experiment_plan` and the `plan_report_renderer` agent emits the report. **Phase 2 — deterministic realization** (`RealizeBoard`, `harness/stages/realize_board.py`): a map→reduce→compile realizer over the frozen board — one codegen self-repair worker (`realize_one_task`, sharing `harness.stages.task_codegen`) per task **in parallel** (full coverage by construction), reduce the greens into a single `workflow_source`/`test_source`, then `MaterializeExecution → CompileWorkflow` (`run_workflow.py --compile-only`, **no** real science; `execution_result` tagged `metadata.mode="compile"`). A task that never greens by the attempt ceiling returns *blocked*; any block persists a durable `intervention_request` then raises `TaskRealizationBlockedError` **before** compile — never auto-reverting to phase 1. The `harness/gateways/` subpackage holds the `AgentGateway` Protocol + `StubAgentGateway` (test stub) + `RouterBackedAgentGateway` (production impl whose `call` dispatches on `AgentCallSpec.call_mode` — degenerate `complete_structured` vs full `stream_agentic` ReAct — driven by `agent.router.Router`); the plan LLM-agent registry (schema + output-kind + system-prompt per `agent_name`) is the single shared `harness.gateways.plan_agents`. Both phases persist raw-before-parsed artifacts with consistent lineage; the `molexp.workflow` engine loads only in executor subprocesses (default `LocalExecutor`; inject `DryRunExecutor` to skip them), never in the harness process. Cross-process suspend/resume is **scope-tagged and durable** (`ApprovalScope` `approval_gate` vs `intervention_request` on the run's `SQLiteApprovalStore`; the shared `services.plan_runtime` resume path rebuilds the scoped session and replays). **Production entry point: `molexp plan` (`cli/plan_cmd.py`) and `POST /plan-tasks`** both drive `PlanOrchestrator` through the one shared `services.plan_runtime.drive_plan_mode` path against a content-addressed Run; artifacts land under `run_dir/artifacts`, events + approvals + lineage in `run_dir/harness.sqlite`.
- Uses: agent (sole edge: `agent.router.Router` Protocol — for LLM dispatch via `RouterBackedAgentGateway`), workflow, workspace. **MUST NOT be imported by them.**
- MUST NOT: import `plugins`, `services`, `server`, `cli`. `pydantic_ai` / `pydantic_graph` must not be transitively loaded when `import molexp.harness` runs (they load lazily through `agent.AgentRunner.run()` only; pydantic_graph survives merely as a transitive install of pydantic-ai, never imported by molexp).
- **Public surface is deliberately small**: `molexp.harness.__all__` = **22 symbols** (locked by `tests/test_harness/test_public_surface.py`): the mode/Stage machinery (`PlanOrchestrator` / `ChatMode` / `chat_loop_config` / `Stage` / `StageRunner` / `HarnessRunContext` / `ModeResult`), stores + executors + gate + gateway (`ArtifactStore` / `FileArtifactStore` / `Executor` / `LocalExecutor` / `DryRunExecutor` / `ApprovalGate` / `AgentGateway`), the doc-cited set (`RouterBackedAgentGateway` / `CapabilityRegistry` / `StageExecutionError` / `SQLiteEventLog` / `SQLiteArtifactLineageStore` / `replay_metadata`), and the **approval-inbox pair** (`ApprovalPendingError` / `SQLiteApprovalStore` — vision-loop-01: gates suspend with a durable pending record in the run's `harness.sqlite`, never a hard failure; grants replay, rejections do not). Stages / schemas / validators import from their subpackages (`harness.stages.*` / `harness.schemas.*`), never from the top level. Plan-owned schema names carry the `Plan` prefix to avoid cross-layer collisions: `PlanTaskIR` / `PlanWorkflowIR` / `PlanValidationReport` / `PlanArtifactRef`. `molexp plan` and `POST /plan-tasks` both drive `PlanOrchestrator` via the shared `services.plan_runtime.drive_plan_mode`; the CLI shows a two-phase banner (interactive planning → deterministic realization). The plan CLI **preflights before any disk write** (model → agent-stack import → credential resolution, via `services.plan_runtime.preflight_plan_router`), so a failed `molexp plan` leaves zero residue. Workflow reconstruction for plan-generated / script runs is **one path**: `harness.workflow_recovery.compiled_workflow_for_run` (CLI `molexp run` and lifecycle capabilities share it). Built-in capability catalogs live under `harness.capabilities/` — `curation` (workspace reorg) and `lifecycle` (`run_execute` / `run_resume` / `run_rerun` / `run_cancel` / `runs_prune`, all side-effect-gated; execute is local-only in v1).

**`molexp.services`** — application-service layer between CLI/server and the four domain layers.
- Owns: `plan_runtime` (`PlanOrchestrator` driving + preflight + `drive_plan_mode` + `materialize` / `persist` / `record` for plan records; scope-tagged durable suspend/resume; handles `ApprovalPendingError` as suspension not failure), `curate_runtime`, `operator_config` (the one `agent.model` loader shared by CLI and server), `agent_task_store`, **`agent_context`** (`build_mount_context` / `resolve_scope_dir` / `mount_session_scope` — the one mount-context builder for `molexp agent` and server session create; lazy-exported at `molexp.services`), **`approval_notify`** (in-process approval-change pub/sub — payload-free SSE pings for `GET /api/approvals/events`; emitters live in services so services never import server). This is where "Python 操作 = UI 操作" is enforced: CLI commands and server routes both call services, never each other.
- Uses: harness, agent, workflow, workspace. MUST NOT: import `server` / `cli` / `plugins` — enforced by `tests/test_services/test_import_guard.py` (AST scan); every lower layer's guard forbids importing `services` back.

## On-disk layout

This layout is the **authoritative experiment-directory spec** — the shape any tool restructuring data into a molexp workspace must produce. Children indices are local conveniences rebuilt on `add_folder` / `remove_folder`. Per-attempt artifacts live under `executions/<exec_id>/`; cross-attempt state stays at the run level. Cross-cutting asset/lineage queries scan the authoritative per-scope `assets.json` manifests directly (`assets.scan`) — there is **no** derived SQLite asset index (the former `catalog/index.sqlite` was removed; see the One-source-of-truth law).

```
workspace_root/
├── workspace.json                # workspace ENTITY metadata
├── project.json                  # children INDEX of projects (derived; child cls snake_case + ".json")
├── workspace.events.sqlite       # run-lifecycle event timeline (default-on, created on first emit; read via read_workspace_events — reading never creates it)
├── cache/<key>.json              # singleton CacheFolder — ws.cache
├── meta.yaml                     # OKF concept marker (type → registry) — every concept dir has one
├── index.md                      # OKF narrative; its markdown links ARE the knowledge graph (out_edges)
├── <agent>/                      # Agent (OKF concept, kind=agent.agent): meta.yaml + flat sessions
│   └── <session>/                #   AgentSession (kind=agent.session): meta.yaml + messages.jsonl (binary)
└── projects/<project_id>/        # container subdir "projects/", dir name = slug, NO prefix
    ├── project.json              # project ENTITY metadata
    ├── experiment.json           # children INDEX of experiments (derived)
    └── experiments/<experiment_id>/   # container "experiments/", dir name = slug, NO prefix
        ├── experiment.json       # experiment ENTITY metadata
        ├── run.json              # children INDEX of runs (derived)
        └── runs/run-<run_id>/    # container "runs/", dir name = "run-" + run_id  (← prefix is mandatory)
            ├── run.json          # identity/provenance: params, config_hash, profile, target (NO status/history — those live in _ops/)
            ├── meta.yaml         # OKF concept marker (type=workspace.run)
            ├── _ops/run.json     # OKF hot-state sidecar (RunOpsState): status, ownership, heartbeat, executions — the read source
            ├── assets.json       # run-scoped asset manifest
            ├── artifacts/        # final products
            ├── cache/            # per-run user-domain cache
            └── executions/<exec_id>/   # exec_id = "exec-" + run_id + optional "-N"
                ├── execution.json
                ├── workflow.json # workflow-layer state + completed node outputs (resume seed source; opaque to workspace EXCEPT one read-only slice: Run.get_result falls back to completed-node outputs)
                ├── stdout.log / stderr.log / error.txt
                ├── logs/<name>.log
                └── jobs/<uuid>/  # molq scheduler manifests
```

**Layout naming law (frozen).** Each `Folder` level obeys four derivations, all driven off the class hierarchy `Workspace → Project → Experiment → Run`:
- **Container subdir** holding a level's children is the child kind pluralized: `projects/`, `experiments/`, `runs/`.
- **Directory name** within that container is the slugified id (kebab) for Project/Experiment, with **no prefix** — except a **Run dir is always prefixed `run-`** (`runs/run-<run_id>/`); the `run-` prefix is part of the contract, not cosmetic.
- **Entity metadata filename** at a level is the level's own class name snake_case + `.json` (`workspace.json` / `project.json` / `experiment.json` / `run.json`) — the single authoritative record for that node.
- **Children-index filename** in a parent dir is the *child* class name snake_case + `.json` (`Folder._index_filename()`), so `<root>/project.json` indexes projects while `<root>/projects/<id>/project.json` is that project's entity file — same basename, different role, never the same directory.

Entity `*.json` and per-scope `assets.json` are **authoritative**; every index (the children-index `*.json`, any future derived view) is **derived** and rebuildable — asset queries scan the manifests directly, with no derived asset index. A Run's hot operational state (status / ownership / heartbeat / executions) lives in its `_ops/run.json` sidecar — the read source — not in the `run.json` entity file. Run ids are 8-char hex or a content-addressed `config_hash`; project/experiment ids are slugs of their names (`add_*` is idempotent on the slug).

Agent-layer mounts (`Agent` / `AgentSession`) are OKF `workspace.Folder` subclasses (registered via `@concept_type("agent.agent"/"agent.session")` against `molexp.knowledge.types`); they can attach at any `Folder` (workspace root, Project, Experiment, or Run) through generic `add_folder`, and `concept_from_dir` reconstructs them from their `meta.yaml` `type` via the shared registry.

## Packaging

`pip install` / `python -m build` **never** invokes npm. The UI is built ahead of time:

```
ui/src/  →  cd ui && npm run build  →  src/molexp/dist/  →  hatchling  →  wheel
```

`src/molexp/dist/` is gitignored (except `.gitkeep`); `create_app()` locates assets via `importlib.resources.files("molexp") / "dist"` and falls back to API-only if empty. Dev mode: backend on `:8000`, frontend on `:5173`.

**UI component library:** prefer **shadcn/ui** when building UI features; if a shadcn component doesn't fit, document the reason in the PR description.

## Data type ownership

Each conceptual data category lives in exactly **one** layer. Cross-layer references flow downward through the public surface of the lower layer — `workflow` imports `workspace.Run` is fine; `workspace` importing `workflow.Workflow` is forbidden. For the full ownership table (concept → owning module) see `.claude/notes/architecture.md`.

**Pydantic vs plain class.** Pure data types (events, configs, results, lineage records, IR nodes) are `pydantic.BaseModel(frozen=True)`. Runtime containers carrying live instances (callables, asyncio objects, services) are plain Python classes with explicit `__init__`. `arbitrary_types_allowed=True` is **forbidden** in `src/molexp/agent/` — anything that needs it is a runtime container by definition.

## Key patterns

- **Topology-driven parallelism** — tasks grouped into levels by the dependency graph; same-level tasks run in parallel automatically.
- **Content-addressed caching** — `cache_key = f(snapshot.key, inputs_hash)` where `snapshot.key == f"{code_hash}:{config_hash}"` (`TaskSnapshot` — AST-normalized code + config; **inputs never fold into the snapshot**) and `inputs_hash` is owned by `Caching` (JSON-canonical over the task `inputs` channel: upstream outputs, sweep params, content-addressed `workdir` Path with stable Path serialization). Execution location is not task identity. Two sweep runs with different params never share a root-task cache entry. Backed by `ws.cache`, never `~/.molexp/cache/`. Locked by `tests/test_workflow/test_cache_contract.py`.
- **Atomic persistence** — all JSON writes use temp-file + `os.rename`; workflow-layer writes go through `workspace.atomic_write_json`.
- **Values-on-edges lowering** — the workflow DAG is lowered to a frozen molexp-owned `ExecutionPlan` (`workflow/_engine/plan.py`) executed by the structural engine (`engine.py` `run_plan`): each task's inputs are delivered from its upstreams' outputs (declared `depends_on` wins; trigger-carried values reach dep-less targets — the loop-back/branch-routed input channel), a task launches exactly when its dependencies are satisfied, and deadlock is detected structurally (`WorkflowDeadlockError`, zero timing constants). The engine is fully self-owned: the pydantic-graph dependency was removed, `End` is molexp's own sentinel (`workflow/types.py`), and no `src/` module may import `pydantic_graph` (full-src AST scan in `tests/test_workflow/test_engine_boundary.py`). A parallel-join and a loop-`until` cannot be fused onto the same task — use separate tasks.
