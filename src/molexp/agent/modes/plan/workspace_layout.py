"""On-disk layout helper for PlanMode-materialized experiment workspaces.

PlanMode emits a structured tree of artifacts — original report, digest,
implementation plan, workflow / task IR, generated source, generated
tests, runtime outputs, plus a manifest and validation report. This
module owns the directory shape and the atomic writes; nothing in
``molexp.workspace`` knows about the layout.

Subsystem reservation: the agent layer owns the kind string
``agent.plan-experiments`` (lowercase ASCII, dot-separated, no path
traversal — matches the workspace charter grammar). Each plan instance
lives at ``<workspace_root>/.subsystems/agent.plan-experiments/<plan_id>/``;
``plan_id`` is generated at first :meth:`PlanWorkspaceHandle.materialize`
and stable for the lifetime of the plan.

Lazy materialization: directory creation happens on first ``*_dir()``
call (mirrors ``SubsystemStore.dir()``); construction itself is
side-effect-free. ``manifest_path()`` and ``validation_report_path()``
return the file paths without creating them — they exist only after
:meth:`PlanWorkspaceHandle.write_manifest` / ``write_validation_report``
runs.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from molexp.workspace import Workspace, atomic_write_text

if TYPE_CHECKING:
    from molexp.agent.modes.plan.schemas import ApprovalDecision

__all__ = [
    "AGENT_PLAN_EXPERIMENTS_KIND",
    "CheckResult",
    "PlanManifest",
    "PlanStatus",
    "PlanWorkspaceHandle",
    "RepairIterationRecord",
    "ValidationReport",
]

AGENT_PLAN_EXPERIMENTS_KIND = "agent.plan-experiments"
"""Subsystem-store kind string reserved for PlanMode-materialized workspaces.

Format follows the workspace charter grammar (lowercase ASCII,
dot-separated, no path traversal). The agent layer owns this kind;
workspace assigns no semantics."""


_FROZEN = ConfigDict(frozen=True, extra="forbid")

PlanStatus = Literal[
    "draft",
    "validated",
    "validation_failed",
    "ready_for_review",
    "approved",
    "approved_with_override",
    "ready_for_run",
    "pending_review",
]
"""PlanMode workspace lifecycle state.

The values intentionally distinguish human approval from machine
readiness. A failed validation pass may still leave a reviewable
workspace, but it must not be marked ``ready_for_run``.
"""


# ── Frozen-pydantic data ───────────────────────────────────────────────────


class CheckResult(BaseModel):
    """One row in a :class:`ValidationReport`'s checks table.

    Attributes:
        name: Human-readable check identifier (e.g. ``"workflow_ir_parseable"``).
        passed: Whether the check passed.
        severity: ``"info"`` / ``"warning"`` / ``"error"``. Only
            ``"error"``-severity failures flip the overall report's
            ``passed`` flag to ``False``.
        detail: Free-form human-readable explanation; empty string if
            the check passed without a note.
    """

    model_config = _FROZEN

    name: str
    passed: bool
    severity: Literal["info", "warning", "error"]
    detail: str = ""


class ValidationReport(BaseModel):
    """Aggregate result of PlanMode's validation pass.

    Rendered to ``validation_report.md`` via
    :meth:`PlanWorkspaceHandle.write_validation_report`.
    """

    model_config = _FROZEN

    passed: bool
    checks: tuple[CheckResult, ...] = ()
    summary: str = ""

    def to_markdown(self) -> str:
        """Render the report as a human-readable markdown document.

        Layout: H1 header reflecting overall ``passed`` state, a brief
        summary paragraph, and a four-column table (``name | severity
        | passed | detail``) with one row per check.
        """
        status_word = "passed" if self.passed else "failed"
        lines = [
            f"# Validation report — {status_word}",
            "",
            self.summary if self.summary else "_(no summary)_",
            "",
            "| name | severity | passed | detail |",
            "| ---- | -------- | ------ | ------ |",
        ]
        for check in self.checks:
            cell_passed = "✓" if check.passed else "✗"
            # Escape pipes in detail so the markdown table stays well-formed.
            detail = check.detail.replace("|", "\\|") or "_(none)_"
            lines.append(f"| {check.name} | {check.severity} | {cell_passed} | {detail} |")
        if not self.checks:
            lines.append("| _(no checks)_ |  |  |  |")
        return "\n".join(lines) + "\n"


class RepairIterationRecord(BaseModel):
    """Per-iteration audit row stored on :attr:`PlanManifest.repair_history`.

    Captures the structured rejection that the human reviewer issued in
    iteration *n* and when the prior live-tree snapshot was archived
    under ``<plan_id>/repairs/iter-<n>/``.

    Lives in this module (not ``schemas.py``) so :class:`PlanManifest`
    can reference it concretely — :mod:`molexp.agent.modes.plan.schemas`
    already imports from this module for :class:`CheckResult`, so
    placing the record here avoids a circular import. The schemas
    module re-exports the class so callers don't need to know the
    physical location.
    """

    model_config = _FROZEN

    iteration: int
    target_node_ids: tuple[str, ...] = ()
    target_task_ids: tuple[str, ...] = ()
    cascade_downstream: bool = False
    archived_at: datetime
    feedback: str = ""


class PlanManifest(BaseModel):
    """On-disk manifest for a single PlanMode experiment workspace.

    Persisted as ``manifest.yaml`` in the plan root.

    Attributes:
        plan_id: Stable UUIDv7 identifier for the plan.
        created_at: UTC timestamp of manifest creation.
        report_source: Opaque identifier for the user-supplied report
            (relative path under ``report/`` or a content hash).
        workflow_ir_path: Path to the materialized workflow IR
            (``ir/workflow.yaml``), relative to the plan root.
        task_ir_paths: Paths to per-task IR files
            (``ir/tasks/<name>.yaml``), relative to the plan root.
        model_policy_snapshot: Serialized form of the active
            :class:`PlanModelPolicy` (sub-spec 04). ``None`` when no
            policy was attached at materialize time. Plain dict to keep
            workspace_layout decoupled from the policy module.
        status: Lifecycle marker. ``"ready_for_run"`` is reserved for
            workspaces that passed the RunMode-style handoff check.
        repair_iterations: Count of completed review→repair rounds.
            Zero on a freshly materialized plan; incremented before each
            repair iteration writes its archive under
            ``<plan_id>/repairs/iter-<n>/``.
        repair_history: Audit tuple — one
            :class:`~molexp.agent.modes.plan.schemas.RepairIterationRecord`
            per past repair round. The element at index ``n`` describes
            the rejection that caused round ``n``.
    """

    model_config = _FROZEN

    plan_id: str
    created_at: datetime
    report_source: str
    workflow_ir_path: Path
    task_ir_paths: tuple[Path, ...] = ()
    model_policy_snapshot: dict[str, Any] | None = None
    status: PlanStatus = "draft"
    repair_iterations: int = 0
    repair_history: tuple[RepairIterationRecord, ...] = ()


# ── Runtime container ──────────────────────────────────────────────────────


class PlanWorkspaceHandle:
    """Stable on-disk handle to one PlanMode experiment workspace.

    Plain Python class (not a pydantic model) because it carries a live
    :class:`Workspace` reference. Construct via the classmethods, never
    via ``__init__`` directly:

    - :meth:`materialize` for a new plan (generates a fresh ``plan_id``).
    - :meth:`attach` for an existing plan_id (resume / re-open).

    The per-plan directory is created lazily on the first
    ``*_dir()``-returning call. Construction is side-effect-free —
    nothing touches disk until a directory or file is requested.
    """

    def __init__(self, workspace: Workspace, plan_id: str) -> None:
        # Public construction goes through ``materialize`` / ``attach``;
        # __init__ is the shared low-level path both classmethods use.
        self._workspace = workspace
        self.plan_id = plan_id

    # -- Construction -----------------------------------------------------

    @classmethod
    def materialize(
        cls,
        workspace: Workspace,
        plan_id: str | None = None,
    ) -> PlanWorkspaceHandle:
        """Build a handle for a fresh plan (or attach to a supplied id).

        Args:
            workspace: Host workspace; must be the same workspace for
                the lifetime of the plan.
            plan_id: Optional explicit identifier. When ``None``, a
                fresh UUIDv7 (or, if ``uuid7`` is unavailable, UUIDv4)
                hex string is generated.

        Returns:
            Handle pointing at
            ``<workspace_root>/.subsystems/agent.plan-experiments/<plan_id>/``.
            The directory itself is **not** created until the first
            ``*_dir()`` call.
        """
        if plan_id is None:
            plan_id = _new_plan_id()
        return cls(workspace, plan_id)

    @classmethod
    def attach(
        cls,
        workspace: Workspace,
        plan_id: str,
    ) -> PlanWorkspaceHandle:
        """Attach to an existing plan id without generating a new one.

        Used by resumed runs. Mirror of :meth:`materialize` minus the
        id-generation branch — this method is the explicit signal to
        readers that the caller intends to address an existing plan.
        """
        return cls(workspace, plan_id)

    # -- Internal helpers ------------------------------------------------

    def _subsystem_root(self) -> Path:
        """Return the kind-private subsystem root (creates it if needed)."""
        return self._workspace.subsystem_store(AGENT_PLAN_EXPERIMENTS_KIND).dir()

    def _ensure(self, segments: tuple[str, ...]) -> Path:
        """Resolve, mkdir, and return the path under the plan root."""
        path = self._subsystem_root().joinpath(self.plan_id, *segments)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve(self, segments: tuple[str, ...]) -> Path:
        """Resolve a path under the plan root without creating anything."""
        return self._subsystem_root().joinpath(self.plan_id, *segments)

    # -- Path API --------------------------------------------------------

    def root(self) -> Path:
        """Path to the plan root directory; created if missing."""
        return self._ensure(())

    def report_dir(self) -> Path:
        """``<plan_id>/report/`` — original + digest report files."""
        return self._ensure(("report",))

    def plan_dir(self) -> Path:
        """``<plan_id>/plan/`` — natural-language implementation plan."""
        return self._ensure(("plan",))

    def ir_dir(self) -> Path:
        """``<plan_id>/ir/`` — workflow + task IR YAML files."""
        return self._ensure(("ir",))

    def tasks_ir_dir(self) -> Path:
        """``<plan_id>/ir/tasks/`` — one YAML file per task IR."""
        return self._ensure(("ir", "tasks"))

    def src_dir(self) -> Path:
        """``<plan_id>/src/`` — generated Python source root."""
        return self._ensure(("src",))

    def experiment_pkg_dir(self) -> Path:
        """``<plan_id>/src/experiment/`` — generated experiment package."""
        return self._ensure(("src", "experiment"))

    def tasks_pkg_dir(self) -> Path:
        """``<plan_id>/src/experiment/tasks/`` — one module per task class."""
        return self._ensure(("src", "experiment", "tasks"))

    def tests_dir(self) -> Path:
        """``<plan_id>/tests/`` — generated contract tests."""
        return self._ensure(("tests",))

    def configs_dir(self) -> Path:
        """``<plan_id>/configs/`` — runtime configuration overlays."""
        return self._ensure(("configs",))

    def runs_dir(self) -> Path:
        """``<plan_id>/runs/`` — RunMode execution artifacts."""
        return self._ensure(("runs",))

    def results_dir(self) -> Path:
        """``<plan_id>/results/`` — final-product summaries."""
        return self._ensure(("results",))

    def manifest_path(self) -> Path:
        """``<plan_id>/manifest.yaml`` — does **not** create the file."""
        return self._resolve(("manifest.yaml",))

    # -- Repair-iteration archive (planmode-review-repair-loop) ---------

    def repairs_dir(self, iteration: int) -> Path:
        """``<plan_id>/repairs/iter-<iteration>/`` — created if missing.

        One directory per completed review→repair round. The contents
        mirror the live ``report/ plan/ ir/ src/ tests/`` subtrees as
        they were *before* this iteration's overwrites began. See
        :meth:`archive_artifacts_for_repair` for the canonical writer.
        """
        return self._ensure(("repairs", f"iter-{iteration}"))

    def latest_decision_path(self) -> Path:
        """``<plan_id>/repairs/latest_decision.yaml`` — does **not** create the file.

        The repair-loop driver writes the most recent
        :class:`~molexp.agent.modes.plan.schemas.ApprovalDecision` here so
        a re-attached driver (or an out-of-band inspection tool) can see
        what the previous gate decided without re-loading the manifest.
        """
        return self._resolve(("repairs", "latest_decision.yaml"))

    def archive_artifacts_for_repair(self, iteration: int) -> None:
        """Snapshot the live ``report/ plan/ ir/ src/ tests/`` subtrees
        into ``<plan_id>/repairs/iter-<iteration>/``.

        Each subdirectory that exists on disk is copied verbatim with
        :func:`shutil.copytree`; missing subdirectories are silently
        skipped (the materialized plan may not have populated every
        live subtree yet — e.g. an early-iteration archive of a plan
        that bailed out before ``GenerateTaskTests``). Non-empty target
        subdirectories under the archive are an error: each iteration's
        archive must be a fresh write.
        """
        archive_root = self.repairs_dir(iteration)
        for sub in ("report", "plan", "ir", "src", "tests"):
            live = self._resolve((sub,))
            if not live.exists() or not live.is_dir():
                continue
            target = archive_root / sub
            shutil.copytree(live, target, dirs_exist_ok=False)

    def validation_report_path(self) -> Path:
        """``<plan_id>/validation_report.md`` — does **not** create the file."""
        return self._resolve(("validation_report.md",))

    def validation_report_data_path(self) -> Path:
        """``<plan_id>/validation_report.yaml`` — structured validation data."""
        return self._resolve(("validation_report.yaml",))

    # -- Atomic writers --------------------------------------------------

    def write_manifest(self, manifest: PlanManifest) -> Path:
        """Serialize ``manifest`` to ``manifest_path()`` as YAML.

        Round-trips through pydantic's ``mode="json"`` dump so paths,
        datetimes, and enums become primitives ``yaml.safe_load`` can
        reconstruct unaided.
        """
        path = self.manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = manifest.model_dump(mode="json")
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
        atomic_write_text(path, text)
        return path

    def write_validation_report(self, report: ValidationReport) -> Path:
        """Render ``report`` to markdown and persist a structured YAML twin."""
        path = self.validation_report_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, report.to_markdown())
        data_path = self.validation_report_data_path()
        text = yaml.safe_dump(
            report.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
        )
        atomic_write_text(data_path, text)
        return path

    def write_latest_decision(self, decision: ApprovalDecision) -> Path:
        """Persist the most recent :class:`ApprovalDecision` to
        ``<plan_id>/repairs/latest_decision.yaml`` so an out-of-band
        reader can recover the rejection without parsing the manifest."""
        path = self.latest_decision_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(
            decision.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
        )
        atomic_write_text(path, text)
        return path

    # -- Generated-source writers (sub-spec 06) ---------------------------

    def write_test_module(self, task_id: str, source: str) -> Path:
        """Write a per-task test module to ``tests/test_<task_id>.py``."""
        path = self.tests_dir() / f"test_{task_id}.py"
        atomic_write_text(path, source)
        return path

    def write_workflow_structure_test(self, source: str) -> Path:
        """Write the topology-pin test to ``tests/test_workflow_structure.py``."""
        path = self.tests_dir() / "test_workflow_structure.py"
        atomic_write_text(path, source)
        return path

    def write_task_implementation(self, task_id: str, source: str) -> Path:
        """Write a per-task module to ``src/experiment/tasks/<task_id>.py``."""
        path = self.tasks_pkg_dir() / f"{task_id}.py"
        atomic_write_text(path, source)
        return path


# ── Helpers ────────────────────────────────────────────────────────────────


def _new_plan_id() -> str:
    """Generate a fresh plan id.

    Uses :func:`uuid.uuid7` on Python 3.14+ (UUIDv7 is monotonic by
    timestamp — friendlier ordering on disk), falling back to
    :func:`uuid.uuid4` otherwise. The hex form is returned (no dashes)
    so the id is filesystem-safe across all platforms.
    """
    factory = getattr(uuid, "uuid7", None)
    if factory is None:
        return uuid.uuid4().hex
    return factory().hex
