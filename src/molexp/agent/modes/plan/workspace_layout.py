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

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from molexp.workspace import Workspace, atomic_write_text

__all__ = [
    "AGENT_PLAN_EXPERIMENTS_KIND",
    "CheckResult",
    "PlanManifest",
    "PlanWorkspaceHandle",
    "ValidationReport",
]

AGENT_PLAN_EXPERIMENTS_KIND = "agent.plan-experiments"
"""Subsystem-store kind string reserved for PlanMode-materialized workspaces.

Format follows the workspace charter grammar (lowercase ASCII,
dot-separated, no path traversal). The agent layer owns this kind;
workspace assigns no semantics."""


_FROZEN = ConfigDict(frozen=True, extra="forbid")


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
        status: Lifecycle marker — ``"draft"`` after initial
            materialize, ``"validated"`` after PlanMode's validation
            pass, ``"approved"`` after human review.
    """

    model_config = _FROZEN

    plan_id: str
    created_at: datetime
    report_source: str
    workflow_ir_path: Path
    task_ir_paths: tuple[Path, ...] = ()
    model_policy_snapshot: dict[str, Any] | None = None
    status: Literal["draft", "validated", "approved"] = "draft"


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

    def validation_report_path(self) -> Path:
        """``<plan_id>/validation_report.md`` — does **not** create the file."""
        return self._resolve(("validation_report.md",))

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
        """Render ``report`` to markdown and write to ``validation_report_path()``."""
        path = self.validation_report_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, report.to_markdown())
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
