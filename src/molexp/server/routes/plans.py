"""Plan-document routes — read the persisted PlanMode artifacts for an experiment.

A "plan" is a content-addressed Run under the experiment that PlanMode populated
with an ``experiment_report`` artifact (the readable experiment plan) plus a
``user_plan`` artifact (the natural-language draft). The ephemeral plan-*task*
runtime (``plan_tasks.py``) tracks in-flight generation; these routes serve the
durable RESULT so the UI's Agents hub can list and read generated plans without
the originating task still being alive in memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from molexp.server.dependencies import get_workspace

if TYPE_CHECKING:
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.workspace import Workspace
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.run import Run

__all__ = ["flat_router", "router"]

router = APIRouter(
    prefix="/projects/{project_id}/experiments/{experiment_id}/plans",
    tags=["plans"],
)

# Workspace-wide plan listing (the Agents hub shows every plan in one place,
# not just one experiment's). Separate prefix → a sibling router.
flat_router = APIRouter(prefix="/plans", tags=["plans"])


class PlanSummaryResponse(BaseModel):
    """One generated plan in the list (its backing run + report title)."""

    runId: str
    title: str
    status: str
    createdAt: str
    hasWorkflow: bool


class PlanListResponse(BaseModel):
    plans: list[PlanSummaryResponse]
    total: int


class WorkspacePlanSummary(BaseModel):
    """A generated plan + the project/experiment it belongs to (workspace-wide)."""

    projectId: str
    experimentId: str
    runId: str
    title: str
    status: str
    createdAt: str
    hasWorkflow: bool


class WorkspacePlanListResponse(BaseModel):
    plans: list[WorkspacePlanSummary]
    total: int


class PlanTaskInfo(BaseModel):
    """One task of the generated workflow (the plan's executable spec)."""

    id: str
    type: str | None = None
    source: str | None = None


class PlanFile(BaseModel):
    """One file of a multi-file generated program (relative path + source)."""

    path: str
    source: str


class PlanDetailResponse(BaseModel):
    """The full generated plan: every deliverable the 9-step pipeline produced.

    One field per UI deliverable view. ``experimentReport`` is the human-readable
    proposal (step 1); ``experimentSpec`` (+ ``experimentSpecYaml``) is the
    concrete spec (step 2); ``capabilities`` is the resolved toolchain catalog
    (step 3); ``tasks`` + ``workflowSource`` are the bound tasks + runnable source
    (steps 4-5); ``inputSet`` is the parameter-space sweep (step 6); ``dryRun`` is
    the compile/dry-run result (step 7); ``executionReport`` is the where/how
    hand-off (step 9). All are ``None`` when the step has not run.
    """

    runId: str
    projectId: str
    experimentId: str
    title: str
    status: str
    draft: str
    experimentReport: dict[str, Any] | None
    experimentSpec: dict[str, Any] | None
    experimentSpecYaml: str | None
    capabilities: str | None
    # The workflow IR (step 4): raw artifact + a curated workflow-spec YAML
    # (inputs, tasks with purpose + typed inputs/outputs, task→task edges).
    workflowIr: dict[str, Any] | None
    workflowIrYaml: str | None
    # The generated workflow spec — every task + the runnable source.
    tasks: list[PlanTaskInfo]
    workflowSource: str | None
    # Per-task source + test files (one module per task + the assembly; one test
    # per task). Empty for single-file plans — then ``workflowSource`` stands.
    workflowFiles: list[PlanFile]
    testFiles: list[PlanFile]
    inputSet: dict[str, Any] | None
    dryRun: dict[str, Any] | None
    planReview: dict[str, Any] | None
    executionReport: dict[str, Any] | None
    # Every harness stage artifact this plan produced (kinds present on disk).
    artifactKinds: list[str]
    hasWorkflow: bool


def _artifacts_root(run: Run) -> Path:
    return Path(run.run_dir) / "artifacts"


def _artifact_store(run: Run) -> FileArtifactStore:
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=_artifacts_root(run))


def _read_json_kind(store: FileArtifactStore, root: Path, kind: str) -> dict[str, Any] | None:
    """Read the latest JSON artifact of ``kind`` as a dict, or None.

    Reads the content-addressed file directly from the store layout
    (``<root>/<kind>/<id>.json``) rather than via ``store.get`` so a workspace
    that was MOVED after its artifacts were written (a stale absolute ``uri`` in
    the ref) still resolves; falls back to the store resolver otherwise.
    """
    ref = store.latest_by_kind(kind)
    if ref is None:
        return None
    direct = root / kind / f"{ref.id}.json"
    raw: str | bytes | None = None
    try:
        raw = direct.read_text()
    except OSError:
        try:
            raw = store.get(ref.id)
        except Exception:
            return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _read_text_kind(store: FileArtifactStore, root: Path, kind: str) -> str | None:
    """Read the latest text artifact of ``kind`` (e.g. ``capability_catalog``)."""
    ref = store.latest_by_kind(kind)
    if ref is None:
        return None
    for path in (root / kind / f"{ref.id}.txt", root / kind / f"{ref.id}.md"):
        try:
            return path.read_text()
        except OSError:
            continue
    try:
        return store.get(ref.id).decode("utf-8")
    except Exception:
        return None


def _spec_to_yaml(
    spec: dict[str, Any] | None, workflow_ir: dict[str, Any] | None = None
) -> str | None:
    """Render the draft spec as ONE human-readable YAML for the panel.

    The draft spec is the comprehensive document: the concrete ExperimentSpec
    PLUS a ``workflow_spec:`` section (the curated workflow IR) when the IR
    exists, so the panel renders it as a single un-split YAML block.
    """
    if spec is None:
        return None
    import yaml

    merged = dict(spec)
    workflow_spec = _workflow_ir_to_spec_dict(workflow_ir)
    if workflow_spec is not None:
        merged["workflow_spec"] = workflow_spec
    return yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)


def _pv_value(pv: object) -> object:
    """The ``.value`` of a ParameterValue dict, or the value itself."""
    return pv.get("value") if isinstance(pv, dict) else pv


def _workflow_ir_to_spec_dict(ir: dict[str, Any] | None) -> dict[str, Any] | None:
    """Curate the workflow_ir artifact into a readable workflow-spec mapping.

    Shows the workflow inputs, each task's purpose + typed inputs/outputs, and
    the task→task edges (the DAG links) — the "workflow spec" view, not a raw
    dump. ``outputs`` carry the IR's declared kind per name (the "type"). Reused
    both as its own ``workflowIrYaml`` deliverable (step 4) and as the
    ``workflow_spec:`` section embedded in the draft-spec YAML (step 2).
    """
    if not ir:
        return None
    spec: dict[str, Any] = {"name": ir.get("name")}
    if ir.get("objective"):
        spec["objective"] = ir["objective"]
    inputs = {
        k: {"value": _pv_value(pv), "source": pv.get("source") if isinstance(pv, dict) else None}
        for k, pv in (ir.get("inputs") or {}).items()
    }
    if inputs:
        spec["inputs"] = inputs
    # Per-task upstream deps, derived from edges (target → [sources]).
    edge_list = [e for e in (ir.get("edges") or []) if isinstance(e, dict)]
    upstream: dict[str, list[str]] = {}
    for e in edge_list:
        upstream.setdefault(str(e.get("target_task_id")), []).append(str(e.get("source_task_id")))
    tasks: list[dict[str, Any]] = []
    for t in ir.get("tasks") or []:
        if not isinstance(t, dict):
            continue
        entry: dict[str, Any] = {"id": t.get("id")}
        if t.get("purpose"):
            entry["purpose"] = t["purpose"]
        if t.get("task_type"):
            entry["task_type"] = t["task_type"]
        # Upstream (what this task consumes) — the "downstream" is the inverse.
        if upstream.get(str(t.get("id"))):
            entry["depends_on"] = upstream[str(t.get("id"))]
        tin = {k: _pv_value(pv) for k, pv in (t.get("inputs") or {}).items()}
        if tin:
            entry["inputs"] = tin
        if t.get("outputs"):
            entry["outputs"] = t["outputs"]
        tasks.append(entry)
    spec["tasks"] = tasks
    edges = [f"{e.get('source_task_id')} → {e.get('target_task_id')}" for e in edge_list]
    if edges:
        spec["dataflow"] = edges
    return spec


def _workflow_ir_to_spec_yaml(ir: dict[str, Any] | None) -> str | None:
    """Render the curated workflow-spec mapping as YAML (the step-4 deliverable)."""
    spec = _workflow_ir_to_spec_dict(ir)
    if spec is None:
        return None
    import yaml

    return yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)


def _report_title(report: dict[str, Any] | None, fallback: str) -> str:
    if report is None:
        return fallback
    title = report.get("title")
    return title if isinstance(title, str) and title.strip() else fallback


def _read_workflow_source(store: FileArtifactStore, root: Path) -> str | None:
    """The generated ``build_workflow()`` Python (from the workflow_source artifact)."""
    data = _read_json_kind(store, root, "workflow_source")
    if data is None:
        return None
    source = data.get("source")
    return source if isinstance(source, str) else None


def _read_program_files(store: FileArtifactStore, root: Path, kind: str) -> list[PlanFile]:
    """The per-task files of a generated program (``workflow_source``/``test_source``).

    Returns the artifact's ``files`` (one module/test per task + assembly) when
    present; for a single-file plan it returns one entry (``{module_name}.py``)
    so the UI renders uniformly. Empty when the artifact is absent.
    """
    data = _read_json_kind(store, root, kind)
    if data is None:
        return []
    files = data.get("files")
    if isinstance(files, list) and files:
        return [
            PlanFile(path=str(f.get("path", "")), source=str(f.get("source", "")))
            for f in files
            if isinstance(f, dict)
        ]
    source = data.get("source")
    name = data.get("module_name") or kind
    return [PlanFile(path=f"{name}.py", source=source)] if isinstance(source, str) else []


def _read_tasks(experiment: Experiment) -> list[PlanTaskInfo]:
    """Every task of the generated workflow, from the experiment's persisted IR."""
    raw = experiment.metadata.workflow_source
    if not isinstance(raw, str) or not raw:
        return []
    try:
        ir = json.loads(raw)
    except (ValueError, TypeError):
        return []
    task_configs = ir.get("task_configs") if isinstance(ir, dict) else None
    if not isinstance(task_configs, list):
        return []
    tasks: list[PlanTaskInfo] = []
    for tc in task_configs:
        if not isinstance(tc, dict) or not isinstance(tc.get("task_id"), str):
            continue
        ttype = tc.get("task_type")
        tsrc = tc.get("source")
        tasks.append(
            PlanTaskInfo(
                id=tc["task_id"],
                type=ttype if isinstance(ttype, str) else None,
                source=tsrc if isinstance(tsrc, str) else None,
            )
        )
    return tasks


def _artifact_kinds(root: Path) -> list[str]:
    """The harness stage artifact kinds this plan produced (from the index dir)."""
    index_dir = root / "_index"
    if not index_dir.is_dir():
        return []
    return sorted(p.stem for p in index_dir.glob("*.json"))


@router.get("", response_model=PlanListResponse)
def list_plans(
    project_id: str,
    experiment_id: str,
    workspace: Workspace = Depends(get_workspace),
) -> PlanListResponse:
    """List the experiment's runs that carry a generated plan (experiment_report)."""
    experiment = workspace.get_project(project_id).get_experiment(experiment_id)
    plans: list[PlanSummaryResponse] = []
    for run in experiment.list_runs():
        root = _artifacts_root(run)
        store = _artifact_store(run)
        report = _read_json_kind(store, root, "experiment_report")
        if report is None:
            continue
        plans.append(
            PlanSummaryResponse(
                runId=run.id,
                title=_report_title(report, run.id),
                status=run.status,
                createdAt=run.metadata.created_at.isoformat(),
                hasWorkflow=store.latest_by_kind("workflow_source") is not None,
            )
        )
    plans.sort(key=lambda p: p.createdAt, reverse=True)  # newest first
    return PlanListResponse(plans=plans, total=len(plans))


@flat_router.get("", response_model=WorkspacePlanListResponse)
def list_all_plans(workspace: Workspace = Depends(get_workspace)) -> WorkspacePlanListResponse:
    """List every generated plan in the active workspace (across all experiments)."""
    plans: list[WorkspacePlanSummary] = []
    for project in workspace.list_projects():
        for experiment in project.list_experiments():
            for run in experiment.list_runs():
                root = _artifacts_root(run)
                store = _artifact_store(run)
                report = _read_json_kind(store, root, "experiment_report")
                if report is None:
                    continue
                plans.append(
                    WorkspacePlanSummary(
                        projectId=project.id,
                        experimentId=experiment.id,
                        runId=run.id,
                        title=_report_title(report, run.id),
                        status=run.status,
                        createdAt=run.metadata.created_at.isoformat(),
                        hasWorkflow=store.latest_by_kind("workflow_source") is not None,
                    )
                )
    plans.sort(key=lambda p: p.createdAt, reverse=True)
    return WorkspacePlanListResponse(plans=plans, total=len(plans))


@router.get("/{run_id}", response_model=PlanDetailResponse)
def get_plan(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace: Workspace = Depends(get_workspace),
) -> PlanDetailResponse:
    """Return one generated plan's draft + structured experiment report."""
    from molexp.workspace.errors import RunNotFoundError

    experiment = workspace.get_project(project_id).get_experiment(experiment_id)
    try:
        run = experiment.get_run(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"plan run {run_id!r} not found") from exc

    root = _artifacts_root(run)
    store = _artifact_store(run)
    report = _read_json_kind(store, root, "experiment_report")
    if report is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"run {run_id!r} has no generated plan (no experiment_report artifact)",
        )
    user_plan = _read_json_kind(store, root, "user_plan") or {}
    draft = user_plan.get("raw_text")
    spec = _read_json_kind(store, root, "experiment_spec")
    workflow_ir = _read_json_kind(store, root, "workflow_ir")
    return PlanDetailResponse(
        runId=run.id,
        projectId=project_id,
        experimentId=experiment_id,
        title=_report_title(report, run.id),
        status=run.status,
        draft=draft if isinstance(draft, str) else "",
        experimentReport=report,
        experimentSpec=spec,
        experimentSpecYaml=_spec_to_yaml(spec, workflow_ir),
        capabilities=_read_text_kind(store, root, "capability_catalog"),
        workflowIr=workflow_ir,
        workflowIrYaml=_workflow_ir_to_spec_yaml(workflow_ir),
        tasks=_read_tasks(experiment),
        workflowSource=_read_workflow_source(store, root),
        workflowFiles=_read_program_files(store, root, "workflow_source"),
        testFiles=_read_program_files(store, root, "test_source"),
        inputSet=_read_json_kind(store, root, "input_set"),
        dryRun=_read_json_kind(store, root, "execution_result"),
        planReview=_read_json_kind(store, root, "plan_review"),
        executionReport=_read_json_kind(store, root, "execution_report"),
        artifactKinds=_artifact_kinds(root),
        hasWorkflow=store.latest_by_kind("workflow_source") is not None,
    )
