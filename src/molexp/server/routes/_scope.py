"""Helpers for resolving on-disk scope directories from ``AssetScope``.

The legacy ``_resolve_scope_dir`` in ``asset.py`` makes assumptions that
do not always match ``Run.run_dir`` (which prefixes ``run-`` to the id).
This module navigates the workspace via its public API, so it always
returns the path that actually exists on disk.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workspace.assets import AssetScope


def resolve_scope_dir(workspace, scope: AssetScope) -> Path | None:
    """Return the on-disk directory for ``scope`` using the public workspace API.

    Returns ``None`` if any segment of the scope cannot be resolved.
    """
    if scope.kind == "workspace":
        return workspace.root

    if not scope.ids:
        return None

    project = workspace.get_project(scope.ids[0])
    if project is None:
        return None
    if scope.kind == "project":
        return project.project_dir

    if len(scope.ids) < 2:
        return None
    experiment = project.get_experiment(scope.ids[1])
    if experiment is None:
        return None
    if scope.kind == "experiment":
        return experiment.experiment_dir

    if scope.kind == "run":
        if len(scope.ids) < 3:
            return None
        run = experiment.get_run(scope.ids[2])
        if run is None:
            return None
        return run.run_dir

    return None


def split_workspace_relpath(workspace, abs_or_rel_path: str) -> Path:
    """Resolve a workspace-relative or absolute path against ``workspace.root``."""
    p = Path(abs_or_rel_path).expanduser()
    root = Path(workspace.root).resolve()
    target = p.resolve() if p.is_absolute() else (root / abs_or_rel_path).resolve()
    target.relative_to(root)  # raises ValueError if outside
    return target
