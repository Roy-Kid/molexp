"""Workspace structure ops — thin, idempotent molexp Folder verbs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from molexp.agent.ops.paths import safe_path
from molexp.agent.ops.protocols import EntityRef, TreeView

if TYPE_CHECKING:
    from molexp.workspace import Workspace

_SKIP = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".scratch",  # agent/.scratch
        "_tasks",  # agent/_tasks (UI task index)
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
    }
)


class WorkspaceStructureOps:
    """StructureOps backed by :class:`~molexp.workspace.Workspace`."""

    def __init__(self, workspace_root: Path) -> None:
        self._root = Path(workspace_root).resolve()

    def _ws(self) -> Workspace:
        from molexp.workspace import Workspace

        return Workspace(self._root)

    def materialize(self, name: str = "workspace") -> EntityRef:
        from molexp.workspace import Workspace

        self._root.mkdir(parents=True, exist_ok=True)
        ws = Workspace(self._root, name=name)
        ws.materialize()
        return EntityRef(
            kind="workspace", id=getattr(ws, "id", name), path=str(ws.resolve()), name=name
        )

    def ensure_project(self, name: str) -> EntityRef:
        ws = self._ws()
        ws.materialize()
        project = ws.add_project(name)
        return EntityRef(
            kind="project",
            id=project.id,
            name=project.name,
            path=str(project.resolve()),
        )

    def ensure_experiment(self, project: str, name: str) -> EntityRef:
        ws = self._ws()
        exp = ws.get_project(project).add_experiment(name)
        return EntityRef(
            kind="experiment",
            id=exp.id,
            name=exp.name,
            path=str(exp.resolve()),
        )

    def ensure_run(
        self,
        project: str,
        experiment: str,
        *,
        params: dict[str, object] | None = None,
        run_id: str | None = None,
    ) -> EntityRef:
        """Create-or-get a Run under *project*/*experiment*.

        When *run_id* is set and the run already exists, returns it (params
        are not rewritten). Otherwise mounts a new pending run ready for
        ``code_write`` / ``code_run`` / ``run_land``.
        """
        from typing import cast

        from molexp._typing import JSONValue

        ws = self._ws()
        proj = ws.get_project(project)
        exp = proj.get_experiment(experiment)
        if run_id is not None:
            try:
                existing = exp.get_run(run_id)
                return EntityRef(
                    kind="run",
                    id=existing.id,
                    name=existing.name,
                    path=str(existing.resolve()),
                )
            except Exception:
                pass
        run = exp.add_run(
            cast("dict[str, JSONValue] | None", params),
            id=run_id,
        )
        return EntityRef(
            kind="run",
            id=run.id,
            name=run.name,
            path=str(run.resolve()),
        )

    def inspect(self, path: str = ".") -> TreeView:
        try:
            target = safe_path(self._root, path)
            if not target.is_dir():
                return TreeView(path=path, error=f"not a directory: {path!r}")
            rows: list[str] = []
            for entry in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name)):
                if entry.name in _SKIP:
                    continue
                rows.append(f"{entry.name}/" if entry.is_dir() else entry.name)
            rel = "." if target == self._root else target.relative_to(self._root).as_posix()
            return TreeView(path=rel, entries=tuple(rows))
        except (ValueError, OSError) as exc:
            return TreeView(path=path, error=str(exc))

    def list_projects(self) -> tuple[EntityRef, ...]:
        ws = self._ws()
        return tuple(
            EntityRef(kind="project", id=p.id, name=p.name, path=str(p.resolve()))
            for p in ws.list_projects()
        )

    def list_experiments(self, project: str) -> tuple[EntityRef, ...]:
        ws = self._ws()
        proj = ws.get_project(project)
        return tuple(
            EntityRef(kind="experiment", id=e.id, name=e.name, path=str(e.resolve()))
            for e in proj.list_experiments()
        )
