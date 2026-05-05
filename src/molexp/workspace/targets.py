"""Compute-target registry helpers — the molexp side of the two-axis model.

A :class:`~molexp.workspace.models.ComputeTarget` is a value object stored on
the workspace's ``WorkspaceMetadata.targets`` list.  This module exposes thin
helpers around that list (list/get/add/remove) and the bridge functions
:func:`to_transport` and :func:`target_run_dir` that turn a target into the
molq objects used at submission time.

The workspace itself is the source of truth — these functions read and write
``workspace.json`` via the workspace's own atomic-write path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molq.options import SshTransportOptions
from molq.transport import LocalTransport, SshTransport, Transport

from .models import ComputeTarget

if TYPE_CHECKING:
    from .run import Run
    from .workspace import Workspace


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------


def list_targets(ws: Workspace) -> list[ComputeTarget]:
    """All compute targets registered on *ws*."""
    return list(ws.metadata.targets)


def get_target(ws: Workspace, name: str) -> ComputeTarget:
    """Return the named target.

    Raises:
        KeyError: if no target with this name is registered.
    """
    for t in ws.metadata.targets:
        if t.name == name:
            return t
    raise KeyError(f"no compute target named {name!r}")


def has_target(ws: Workspace, name: str) -> bool:
    """True if a target with *name* is registered."""
    return any(t.name == name for t in ws.metadata.targets)


def add_target(ws: Workspace, target: ComputeTarget) -> None:
    """Register *target* on the workspace.

    Raises:
        ValueError: if a target with the same name already exists.
    """
    if has_target(ws, target.name):
        raise ValueError(f"compute target {target.name!r} already exists")
    ws.metadata = ws.metadata.model_copy(update={"targets": [*ws.metadata.targets, target]})
    ws.save()


def remove_target(ws: Workspace, name: str) -> None:
    """Remove the named target from the workspace registry.

    Raises:
        KeyError: if no target with this name is registered.
    """
    if not has_target(ws, name):
        raise KeyError(f"no compute target named {name!r}")
    ws.metadata = ws.metadata.model_copy(
        update={"targets": [t for t in ws.metadata.targets if t.name != name]}
    )
    ws.save()


# ---------------------------------------------------------------------------
# Bridge to molq
# ---------------------------------------------------------------------------


def to_transport(target: ComputeTarget) -> Transport:
    """Build the molq Transport implied by *target*'s host axis.

    ``host is None`` → :class:`~molq.transport.LocalTransport`.
    Anything else → :class:`~molq.transport.SshTransport` configured from the
    target's ssh options.
    """
    if target.host is None:
        return LocalTransport()
    return SshTransport(
        options=SshTransportOptions(
            host=target.host,
            port=target.port,
            identity_file=target.identity_file,
            ssh_opts=tuple(target.ssh_opts),
        ),
    )


def target_run_dir(target: ComputeTarget, ws: Workspace, run: Run) -> str:
    """Working directory for *run* on *target*'s filesystem.

    For **local targets**, the run dir is already an absolute path on this
    machine — we just return it.  ``scratch_root`` is informational for local
    targets (e.g. shown in `molexp target list`) but doesn't relocate the
    workspace's own directory tree, which would only confuse `molexp explore`
    and break asset paths.

    For **remote targets**, return an absolute path under
    ``target.scratch_root`` namespaced by workspace / project / experiment /
    run ids so concurrent runs on the same target don't collide.
    """
    if not target.is_remote:
        return str(run.run_dir)
    project = run.experiment.project
    experiment = run.experiment
    return (
        f"{target.scratch_root.rstrip('/')}/{ws.metadata.id}/{project.id}/{experiment.id}/{run.id}"
    )
