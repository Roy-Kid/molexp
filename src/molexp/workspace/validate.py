"""Validate a workspace tree against the frozen layout + OKF laws.

Read-only. The layout law lives in one place — the ``Folder`` family and the
on-disk contract it derives (container subdir, the mandatory ``run-`` prefix,
entity vs children-index filenames, the per-concept ``meta.yaml`` marker).
This module is that law expressed as a checker, so a tree assembled by hand,
by an adoption tool, or by an older molexp can be held to the same standard
the writers obey.

The checker answers one question — *does this tree conform?* — and never
repairs. Every finding carries a stable dotted ``rule`` id so callers can
filter, and a severity so a caller can distinguish a broken tree from a
merely incomplete one:

* ``error``   — the layout law is violated; readers may mis-resolve the tree.
* ``warning`` — legal but lazily-created state is absent (a Run that has
  never executed has no ``_ops/run.json`` yet, which is normal).

Derived indexes are checked *against* the authoritative entity dirs, never
the other way round: the One-source-of-truth law makes ``<child>.json`` a
rebuildable cache, so a disagreement is always the index's fault.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from .fs_local import LocalFileSystem

if TYPE_CHECKING:
    from pathlib import Path

    from .fs import FileSystem, PathArg

Severity = Literal["error", "warning"]

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

META_YAML = "meta.yaml"
RUN_DIR_PREFIX = "run-"

#: Structural container subdirs per level — directories that hold children or
#: payload rather than being Concepts themselves, so they carry no meta.yaml.
_CONTAINERS: dict[str, frozenset[str]] = {
    "workspace": frozenset({"projects", "assets", "cache"}),
    "project": frozenset({"experiments", "assets", "cache"}),
    "experiment": frozenset({"runs", "assets", "cache"}),
    "run": frozenset(
        {"artifacts", "assets", "cache", "executions", "metrics", "logs", "jobs", "_ops"}
    ),
}

#: level -> (entity filename, container subdir, children-index filename)
_ENTITY_FILE: dict[str, str] = {
    "workspace": "workspace.json",
    "project": "project.json",
    "experiment": "experiment.json",
    "run": "run.json",
}
_CHILD_OF: dict[str, str] = {
    "workspace": "project",
    "project": "experiment",
    "experiment": "run",
}


class Violation(BaseModel):
    """One conformance finding, anchored at a workspace-relative path."""

    model_config = ConfigDict(frozen=True)

    path: str
    rule: str
    detail: str
    severity: Severity = "error"


class ValidationReport(BaseModel):
    """The outcome of :func:`validate_workspace`."""

    model_config = ConfigDict(frozen=True)

    root: str
    violations: tuple[Violation, ...] = ()

    @property
    def errors(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity == "error")

    @property
    def warnings(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity == "warning")

    @property
    def ok(self) -> bool:
        """True when nothing violates the layout law (warnings are allowed)."""
        return not self.errors

    def summary(self) -> str:
        """One line fit for a CLI or a log."""
        if self.ok and not self.warnings:
            return f"{self.root}: conforms"
        return f"{self.root}: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"


class _Checker:
    """Walks the four tiers once, collecting violations."""

    def __init__(self, root: str, fs: FileSystem) -> None:
        self._root = root
        self._fs = fs
        self._found: list[Violation] = []

    # -- helpers ---------------------------------------------------------

    def _rel(self, path: str) -> str:
        prefix = self._root.rstrip("/") + "/"
        return path[len(prefix) :] if path.startswith(prefix) else path

    def _add(self, path: str, rule: str, detail: str, severity: Severity = "error") -> None:
        self._found.append(
            Violation(path=self._rel(path) or ".", rule=rule, detail=detail, severity=severity)
        )

    def _subdirs(self, path: str) -> list[str]:
        if not self._fs.is_dir(path):
            return []
        names = sorted(self._fs.listdir(path))
        return [
            n for n in names if not n.startswith(".") and self._fs.is_dir(self._fs.join(path, n))
        ]

    def _read_index(self, path: str) -> dict[str, object] | None:
        """Parse a children-index file; report and return None when unreadable."""
        try:
            payload = json.loads(self._fs.read_text(path))
        except (OSError, ValueError) as exc:
            self._add(path, "index.unreadable", f"{type(exc).__name__}: {exc}")
            return None
        if not isinstance(payload, dict):
            self._add(path, "index.malformed", "children index must be a JSON object")
            return None
        return payload

    # -- per-level checks ------------------------------------------------

    def _check_concept(self, path: str, level: str) -> None:
        """Entity file + OKF marker for one concept directory."""
        entity = _ENTITY_FILE[level]
        if not self._fs.is_file(self._fs.join(path, entity)):
            self._add(path, f"{level}.entity", f"missing {entity}")
        if not self._fs.is_file(self._fs.join(path, META_YAML)):
            self._add(path, "concept.marker", f"missing {META_YAML} concept marker")

    def _check_strays(self, path: str, level: str) -> None:
        """Every child dir is a known container or a Concept (has meta.yaml)."""
        allowed = _CONTAINERS[level]
        for name in self._subdirs(path):
            if name in allowed:
                continue
            child = self._fs.join(path, name)
            if self._fs.is_file(self._fs.join(child, META_YAML)):
                continue  # a Concept may mount at any Folder
            self._add(
                child,
                "layout.stray",
                f"{name!r} is neither a container {sorted(allowed)} nor a Concept (no {META_YAML})",
            )

    def _check_index(self, path: str, level: str, child_dirs: list[str]) -> None:
        """The derived children index must match the entity dirs on disk."""
        child_level = _CHILD_OF[level]
        index_name = _ENTITY_FILE[child_level]
        index_path = self._fs.join(path, index_name)

        on_disk = {d[len(RUN_DIR_PREFIX) :] if child_level == "run" else d for d in child_dirs}
        if not self._fs.is_file(index_path):
            if on_disk:
                self._add(
                    path,
                    f"{level}.index",
                    f"missing children index {index_name} for {len(on_disk)} {child_level}(s)",
                )
            return

        payload = self._read_index(index_path)
        if payload is None:
            return
        indexed = set(payload)
        if missing := sorted(on_disk - indexed):
            self._add(
                index_path,
                "index.stale",
                f"{child_level}(s) on disk but absent from the index: {missing}",
            )
        if extra := sorted(indexed - on_disk):
            self._add(
                index_path,
                "index.stale",
                f"{child_level}(s) indexed but absent from disk: {extra}",
            )

    def _check_run(self, path: str) -> None:
        self._check_concept(path, "run")
        self._check_strays(path, "run")
        if not self._fs.is_file(self._fs.join(path, "_ops", "run.json")):
            self._add(
                path,
                "run.ops",
                "no _ops/run.json hot-state sidecar (normal for a run that never executed)",
                severity="warning",
            )

    # -- entry point -----------------------------------------------------

    def run(self) -> list[Violation]:
        root = self._root
        if not self._fs.is_dir(root):
            self._add(root, "workspace.missing", "not a directory")
            return self._found
        if not self._fs.is_file(self._fs.join(root, _ENTITY_FILE["workspace"])):
            self._add(root, "workspace.entity", "missing workspace.json — not a workspace root")
            return self._found

        self._check_concept(root, "workspace")
        self._check_strays(root, "workspace")

        projects_dir = self._fs.join(root, "projects")
        project_dirs = self._subdirs(projects_dir)
        self._check_index(root, "workspace", project_dirs)

        for pname in project_dirs:
            pdir = self._fs.join(projects_dir, pname)
            if not _SLUG_RE.match(pname):
                self._add(pdir, "project.slug", f"{pname!r} is not a kebab-case slug")
            self._check_concept(pdir, "project")
            self._check_strays(pdir, "project")

            experiments_dir = self._fs.join(pdir, "experiments")
            experiment_dirs = self._subdirs(experiments_dir)
            self._check_index(pdir, "project", experiment_dirs)

            for ename in experiment_dirs:
                edir = self._fs.join(experiments_dir, ename)
                if not _SLUG_RE.match(ename):
                    self._add(edir, "experiment.slug", f"{ename!r} is not a kebab-case slug")
                self._check_concept(edir, "experiment")
                self._check_strays(edir, "experiment")

                runs_dir = self._fs.join(edir, "runs")
                run_dirs = self._subdirs(runs_dir)
                self._check_index(edir, "experiment", run_dirs)

                for rname in run_dirs:
                    rdir = self._fs.join(runs_dir, rname)
                    if not rname.startswith(RUN_DIR_PREFIX):
                        self._add(
                            rdir,
                            "run.prefix",
                            f"{rname!r} must be prefixed {RUN_DIR_PREFIX!r}",
                        )
                    self._check_run(rdir)

        return self._found


def validate_workspace(root: PathArg | Path, *, fs: FileSystem | None = None) -> ValidationReport:
    """Check *root* against the workspace layout + OKF laws. Writes nothing.

    Args:
        root: The workspace root directory.
        fs: Filesystem to read through; defaults to the local one, so a remote
            workspace validates over its own transport.

    Returns:
        A :class:`ValidationReport`. ``report.ok`` is True when no ``error``
        was found; warnings never make a tree non-conforming.
    """
    filesystem = fs or LocalFileSystem()
    resolved = filesystem.resolve(root)
    violations = _Checker(resolved, filesystem).run()
    return ValidationReport(root=resolved, violations=tuple(violations))
