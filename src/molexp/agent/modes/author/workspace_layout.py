"""On-disk layout helper for AuthorMode's materialized experiment workspace.

AuthorMode persists every materialization artefact under the *same*
:class:`~molexp.agent.modes.plan.plan_folder.PlanFolder` PlanMode
created — the approved plan and the materialized output live under one
anchor (so RunMode can mount ``runs/`` beside them). :class:`MaterializedLayout`
wraps a bound ``PlanFolder`` and adds the materialization sub-tree the
read-only ``PlanFolder`` deliberately does not own:

    <plan_folder>/
    ├── ir/workflow.yaml          # the lowered workflow IR
    ├── ir/tasks/<task>.yaml      # per-task IR
    ├── src/experiment/           # generated package
    │   ├── __init__.py
    │   ├── workflow.py
    │   └── tasks/<task>.py
    ├── tests/test_<task>.py      # per-task tests
    ├── manifest.yaml             # materialization manifest
    ├── validation_report.json    # ValidateWorkspace verdict
    └── repairs/<id>.json         # per-repair PlanDiff records

Every write goes through :func:`~molexp.workspace.atomic_write_text` for
crash safety. Plain runtime class — it holds a live ``PlanFolder`` handle.
"""

from __future__ import annotations

from pathlib import Path

from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.workspace import atomic_write_text

__all__ = ["MaterializedLayout"]


class MaterializedLayout:
    """Materialization sub-tree helper anchored to a bound :class:`PlanFolder`."""

    def __init__(self, plan_folder: PlanFolder) -> None:
        self._plan_folder = plan_folder

    @property
    def plan_folder(self) -> PlanFolder:
        """The bound :class:`PlanFolder` this layout writes under."""
        return self._plan_folder

    @property
    def plan_id(self) -> str:
        """The plan id (the ``PlanFolder`` name)."""
        return self._plan_folder.plan_id

    # ── directory anchors ────────────────────────────────────────────────

    def root(self) -> Path:
        """Resolve + mkdir the plan-folder root."""
        path = Path(self._plan_folder.path())
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ir_dir(self) -> Path:
        """The ``ir/`` directory (workflow + per-task IR)."""
        return self._ensure(self.root() / "ir")

    def tasks_ir_dir(self) -> Path:
        """The ``ir/tasks/`` directory (per-task IR YAML)."""
        return self._ensure(self.ir_dir() / "tasks")

    def src_dir(self) -> Path:
        """The ``src/`` root of the generated tree."""
        return self._ensure(self.root() / "src")

    def package_dir(self) -> Path:
        """The generated experiment package directory (``src/experiment/``)."""
        return self._ensure(self.src_dir() / "experiment")

    def tasks_pkg_dir(self) -> Path:
        """The generated per-task package (``src/experiment/tasks/``)."""
        return self._ensure(self.package_dir() / "tasks")

    def tests_dir(self) -> Path:
        """The generated ``tests/`` directory."""
        return self._ensure(self.root() / "tests")

    def repairs_dir(self) -> Path:
        """The ``repairs/`` directory (per-repair PlanDiff records)."""
        return self._ensure(self.root() / "repairs")

    @staticmethod
    def _ensure(path: Path) -> Path:
        """Create ``path`` (and parents) if absent and return it."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── named artefact paths ─────────────────────────────────────────────

    def workflow_yaml_path(self) -> Path:
        """Path of the lowered workflow IR (``ir/workflow.yaml``)."""
        return self.ir_dir() / "workflow.yaml"

    def task_ir_path(self, task_id: str) -> Path:
        """Path of one task's IR (``ir/tasks/<task>.yaml``)."""
        return self.tasks_ir_dir() / f"{task_id}.yaml"

    def workflow_py_path(self) -> Path:
        """Path of the generated workflow module (``src/experiment/workflow.py``)."""
        return self.package_dir() / "workflow.py"

    def task_impl_path(self, task_id: str) -> Path:
        """Path of one task's implementation (``src/experiment/tasks/<task>.py``)."""
        return self.tasks_pkg_dir() / f"{task_id}.py"

    def task_test_path(self, task_id: str) -> Path:
        """Path of one task's test (``tests/test_<task>.py``)."""
        return self.tests_dir() / f"test_{task_id}.py"

    def manifest_path(self) -> Path:
        """Path of the materialization manifest (``manifest.yaml``)."""
        return self.root() / "manifest.yaml"

    def validation_report_path(self) -> Path:
        """Path of the workspace validation report (``validation_report.json``)."""
        return self.root() / "validation_report.json"

    def repair_path(self, repair_id: str) -> Path:
        """Path of one repair record (``repairs/<id>.json``)."""
        return self.repairs_dir() / f"{repair_id}.json"

    # ── writers ──────────────────────────────────────────────────────────

    def write(self, path: Path, content: str) -> Path:
        """Atomic-write ``content`` to ``path`` (creating parents)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, content)
        return path
