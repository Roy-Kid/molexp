"""On-disk layout for PlanMode-materialized plan workspaces.

:class:`PlanFolder` is the agent-layer :class:`molexp.workspace.Folder`
subclass owned by PlanMode (``kind = "agent.plan"``). It mounts on any
workspace ``Folder`` via the generic ``add_folder`` API; the plan id
becomes the folder name. The agent layer owns the layout entirely —
workspace stays unaware of the ``agent.plan`` kind.

Mount points::

    ws = Workspace("./lab")
    pf = ws.add_folder(PlanFolder(name="my-plan"))     # workspace-level
    # or under an experiment:
    exp = ws.add_project("proj").add_experiment("exp")
    pf = exp.add_folder(PlanFolder(name="my-plan"))

Subtree layout::

    <parent>/plans/<plan_id>/
    ├── plan.json                  # PlanFolder own metadata (Folder lifecycle)
    ├── manifest.yaml              # PlanManifest (status, history)
    ├── report/                    # original + digest report files
    ├── plan/                      # natural-language implementation plan
    ├── ir/                        # workflow + per-task IR (YAML)
    │   └── tasks/
    ├── src/                       # generated source root
    │   └── experiment/
    │       └── tasks/
    ├── tests/                     # generated contract tests
    ├── configs/                   # runtime config overlays
    ├── capability/                # capability needs + evidence + missing
    ├── validation_report.md       # human-readable validation report
    ├── validation_report.yaml     # structured twin
    ├── repairs/                   # one snapshot dir per review→repair round
    │   └── iter-<n>/
    └── runs/                      # workspace ``Run`` mounts (RunMode hand-off)

Construction is side-effect free; ``parent.add_folder(plan)`` /
:meth:`Folder.path` create directories lazily.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict

from molexp._typing import JSONValue
from molexp.workspace import Folder, FolderMetadata, atomic_write_text
from molexp.workspace.base import _load_metadata, _reconstruct, _save_metadata

if TYPE_CHECKING:
    from molexp.agent.modes.plan.capability import (
        CapabilityEvidenceBatch,
        CapabilityNeedReport,
        MissingCapability,
    )
    from molexp.agent.review import ReviewDecision
    from molexp.workspace.run import Run

__all__ = [
    "AGENT_PLAN_KIND",
    "CheckResult",
    "PlanFolder",
    "PlanFolderMetadata",
    "PlanManifest",
    "PlanStatus",
    "RepairIterationRecord",
    "ValidationReport",
]

AGENT_PLAN_KIND = "agent.plan"
"""Folder ``kind`` for a single PlanMode plan workspace."""

PLAN_METADATA_FILENAME = "plan_folder.json"
"""Per-:class:`PlanFolder` metadata file (auto-derived from class name)."""


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
"""PlanMode plan lifecycle state."""


# ── Frozen-pydantic data ───────────────────────────────────────────────────


class CheckResult(BaseModel):
    """One row in a :class:`ValidationReport`'s checks table."""

    model_config = _FROZEN

    name: str
    passed: bool
    severity: Literal["info", "warning", "error"]
    detail: str = ""


class ValidationReport(BaseModel):
    """Aggregate result of PlanMode's validation pass."""

    model_config = _FROZEN

    passed: bool
    checks: tuple[CheckResult, ...] = ()
    summary: str = ""

    def to_markdown(self) -> str:
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
            detail = check.detail.replace("|", "\\|") or "_(none)_"
            lines.append(f"| {check.name} | {check.severity} | {cell_passed} | {detail} |")
        if not self.checks:
            lines.append("| _(no checks)_ |  |  |  |")
        return "\n".join(lines) + "\n"


class RepairIterationRecord(BaseModel):
    """Per-iteration audit row stored on :attr:`PlanManifest.repair_history`."""

    model_config = _FROZEN

    iteration: int
    target_steps: tuple[str, ...] = ()
    target_task_ids: tuple[str, ...] = ()
    cascade_downstream: bool = False
    archived_at: datetime
    feedback: str = ""


class PlanManifest(BaseModel):
    """On-disk manifest for a single PlanMode plan workspace.

    Persisted as ``manifest.yaml`` in the plan root.
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


class PlanFolderMetadata(FolderMetadata):
    """Frozen lifecycle metadata for a :class:`PlanFolder`.

    Extends :class:`FolderMetadata` so the workspace layer treats it as
    opaque ``Folder`` metadata. The agent layer pins ``kind`` to
    :data:`AGENT_PLAN_KIND` at construction.
    """

    model_config = ConfigDict(frozen=True)


# ── PlanFolder (Folder subclass) ───────────────────────────────────────────


class PlanFolder(Folder):
    """One PlanMode plan workspace — ``kind = "agent.plan"``.

    Construct with an explicit ``name`` (the plan id, slug-safe) or leave
    it to default to a fresh UUIDv7 hex string. Mount via the generic
    :meth:`Folder.add_folder` on any workspace folder::

        plan = ws.add_folder(PlanFolder(name="my-plan"))
        plan.report_dir()  # mkdirs <root>/plans/my-plan/report on demand
    """

    def __init__(
        self,
        *,
        parent: Folder | None = None,
        name: str | None = None,
        kind: str = AGENT_PLAN_KIND,
        _entity_metadata: PlanFolderMetadata | None = None,
    ) -> None:
        plan_id = name if name is not None else _new_plan_id()
        super().__init__(parent=parent, name=plan_id, kind=kind)
        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else PlanFolderMetadata(id=self._name, name=plan_id, kind=kind)
        )
        self._entity_metadata: PlanFolderMetadata = meta

    # ── Folder hooks ─────────────────────────────────────────────────────

    def _compute_path(self) -> Path:
        if self._parent is None:
            raise RuntimeError(
                f"PlanFolder {self._name!r} is unmounted — mount via parent.add_folder()"
            )
        return type(self)._child_dir(self._parent, self._name)

    @classmethod
    def _child_dir(cls, parent: Folder, derived_id: str) -> Path:
        """Plans live under ``<parent>/plans/<plan_id>/``."""
        return parent.path() / "plans" / derived_id

    @classmethod
    def _from_disk(cls, child_dir: Path, parent: Folder) -> PlanFolder:
        meta = _load_metadata(PlanFolderMetadata, child_dir / PLAN_METADATA_FILENAME)
        return _reconstruct(
            cls,
            {
                "_parent": parent,
                "_name": meta.id,
                "_kind": AGENT_PLAN_KIND,
                "_root_path": None,
                "_metadata": FolderMetadata(
                    id=meta.id,
                    name=meta.name,
                    kind=AGENT_PLAN_KIND,
                    created_at=meta.created_at,
                    updated_at=meta.updated_at,
                ),
                "_children_cache": {},
                "_entity_metadata": meta,
            },
        )

    def materialize(self) -> None:
        self.path().mkdir(parents=True, exist_ok=True)
        _save_metadata(self._entity_metadata, self.path() / PLAN_METADATA_FILENAME)

    def save(self) -> None:
        _save_metadata(self._entity_metadata, self.path() / PLAN_METADATA_FILENAME)

    def _to_index_row(self) -> dict[str, JSONValue]:
        return cast("dict[str, JSONValue]", self._entity_metadata.model_dump(mode="json"))

    @property
    def metadata(self) -> PlanFolderMetadata:  # type: ignore[override]
        return self._entity_metadata

    @property
    def plan_id(self) -> str:
        """Alias for :attr:`Folder.name` for legacy plan-mode call sites."""
        return self._name

    # ── Internal directory helper ────────────────────────────────────────

    def _ensure(self, *segments: str) -> Path:
        """Resolve, mkdir, and return a path under this plan."""
        path = self.path().joinpath(*segments)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve(self, *segments: str) -> Path:
        """Resolve a path under this plan without creating anything.

        Falls back to ``self._compute_path()`` so reads on unmaterialized
        plans don't side-effect a mkdir.
        """
        return self._compute_path().joinpath(*segments)

    # ── Path API ─────────────────────────────────────────────────────────

    def root(self) -> Path:
        """Path to the plan root directory; mkdirs if missing."""
        return self.path()

    def report_dir(self) -> Path:
        """``<plan>/report/`` — original + digest report files."""
        return self._ensure("report")

    def plan_dir(self) -> Path:
        """``<plan>/plan/`` — natural-language implementation plan."""
        return self._ensure("plan")

    def ir_dir(self) -> Path:
        """``<plan>/ir/`` — workflow + task IR YAML files."""
        return self._ensure("ir")

    def tasks_ir_dir(self) -> Path:
        """``<plan>/ir/tasks/`` — one YAML file per task IR."""
        return self._ensure("ir", "tasks")

    def src_dir(self) -> Path:
        """``<plan>/src/`` — generated Python source root."""
        return self._ensure("src")

    def experiment_pkg_dir(self) -> Path:
        """``<plan>/src/experiment/`` — generated experiment package."""
        return self._ensure("src", "experiment")

    def tasks_pkg_dir(self) -> Path:
        """``<plan>/src/experiment/tasks/`` — one module per task class."""
        return self._ensure("src", "experiment", "tasks")

    def tests_dir(self) -> Path:
        """``<plan>/tests/`` — generated contract tests."""
        return self._ensure("tests")

    def configs_dir(self) -> Path:
        """``<plan>/configs/`` — runtime configuration overlays."""
        return self._ensure("configs")

    def capability_dir(self) -> Path:
        """``<plan>/capability/`` — capability needs / evidence / misses."""
        return self._ensure("capability")

    def manifest_path(self) -> Path:
        """``<plan>/manifest.yaml`` — does **not** create the file."""
        return self._resolve("manifest.yaml")

    # ── Repair-iteration archive ─────────────────────────────────────────

    def repairs_dir(self, iteration: int) -> Path:
        """``<plan>/repairs/iter-<iteration>/`` — created if missing."""
        return self._ensure("repairs", f"iter-{iteration}")

    def latest_decision_path(self) -> Path:
        """``<plan>/repairs/latest_decision.yaml`` — does **not** create the file."""
        return self._resolve("repairs", "latest_decision.yaml")

    def archive_artifacts_for_repair(self, iteration: int) -> None:
        """Snapshot the live ``report/ plan/ ir/ src/ tests/`` subtrees
        into ``<plan>/repairs/iter-<iteration>/``."""
        archive_root = self.repairs_dir(iteration)
        for sub in ("report", "plan", "ir", "src", "tests"):
            live = self._resolve(sub)
            if not live.exists() or not live.is_dir():
                continue
            target = archive_root / sub
            shutil.copytree(live, target, dirs_exist_ok=False)

    def validation_report_path(self) -> Path:
        """``<plan>/validation_report.md`` — does **not** create the file."""
        return self._resolve("validation_report.md")

    def validation_report_data_path(self) -> Path:
        """``<plan>/validation_report.yaml`` — structured validation data."""
        return self._resolve("validation_report.yaml")

    # ── Atomic writers ───────────────────────────────────────────────────

    def write_manifest(self, manifest: PlanManifest) -> Path:
        """Serialize ``manifest`` to ``manifest_path()`` as YAML."""
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

    def write_latest_decision(self, decision: ReviewDecision) -> Path:
        """Persist the most recent :class:`ReviewDecision` to
        ``<plan>/repairs/latest_decision.yaml``."""
        path = self.latest_decision_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(
            decision.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
        )
        atomic_write_text(path, text)
        return path

    # ── Capability writers (Phase 4) ────────────────────────────────────

    def write_capability_needs(self, report: CapabilityNeedReport) -> Path:
        """Persist ``DraftCapabilityNeeds`` output to ``capability/needs.yaml``."""
        path = self.capability_dir() / "needs.yaml"
        text = yaml.safe_dump(
            report.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
        )
        atomic_write_text(path, text)
        return path

    def write_capability_evidence(self, batch: CapabilityEvidenceBatch) -> Path:
        """Persist ``DiscoverCapabilities`` output to ``capability/evidence.yaml``."""
        path = self.capability_dir() / "evidence.yaml"
        text = yaml.safe_dump(
            batch.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
        )
        atomic_write_text(path, text)
        return path

    def write_capability_missing(self, missing: tuple[MissingCapability, ...]) -> Path:
        """Render the missing-capability ledger to ``capability/missing.md``."""
        path = self.capability_dir() / "missing.md"
        lines: list[str] = ["# Missing capabilities", ""]
        if not missing:
            lines.append("_(none)_")
        else:
            lines.extend(
                [
                    "| reason | need | repairable | detail |",
                    "| ------ | ---- | ---------- | ------ |",
                ]
            )
            for entry in missing:
                if entry.need is not None:
                    need_cell = f"{entry.need.task_id}: {entry.need.capability}"
                else:
                    need_cell = "_(no need)_"
                cell_repairable = "yes" if entry.repairable else "no"
                detail = entry.detail.replace("|", "\\|") or "_(none)_"
                lines.append(f"| {entry.reason} | {need_cell} | {cell_repairable} | {detail} |")
        atomic_write_text(path, "\n".join(lines) + "\n")
        return path

    # ── Generated-source writers ─────────────────────────────────────────

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

    # ── Run hand-off (workspace ``Run`` semantic sugar) ──────────────────

    def add_run(self, name: str, **kwargs: Any) -> Run:
        """Mount a workspace :class:`Run` under this plan for RunMode hand-off.

        Plans hand off to RunMode by allocating an idempotent run-slug
        directory under ``<plan>/runs/<run_id>/``. This is the
        workspace ``Run`` Folder mounted via the generic ``add_folder``
        machinery — RunMode then drives it just like any user-launched
        run.
        """
        from molexp.workspace.run import Run

        run = Run(parent=self, name=name, **kwargs)
        return cast("Run", self.add_folder(run))

    def get_run(self, name: str) -> Run:
        from molexp.workspace.run import Run

        return self.get_folder(name, cls=Run)

    def has_run(self, name: str) -> bool:
        from molexp.workspace.run import Run

        return self.has_folder(name, cls=Run)

    def list_runs(self) -> list[Run]:
        from molexp.workspace.run import Run

        return self.list_folders(cls=Run)

    def remove_run(self, name: str) -> None:
        from molexp.workspace.run import Run

        self.remove_folder(name, cls=Run)


# ── Helpers ────────────────────────────────────────────────────────────────


def _new_plan_id() -> str:
    """Generate a fresh plan id (UUIDv7 hex on 3.14+, UUIDv4 otherwise)."""
    factory = getattr(uuid, "uuid7", None)
    if factory is None:
        return uuid.uuid4().hex
    return factory().hex
