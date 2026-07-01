"""``resolve_object_ref`` — map an :class:`ObjectRef` to a live workspace entity.

integration.md §8.2. A granted :class:`~molexp.harness.schemas.change_proposal.ChangeProposal`
names the objects it may touch as ``ObjectRef(kind, id)``; a curation handler
needs the *live* workspace entity behind each ref to actually mutate it. This is
the single owner of that resolution, over the :mod:`molexp.workspace` public API
only (the ``harness → workspace`` edge is legal).

An unresolvable ref (missing id, unknown kind) fails loudly with
:class:`ObjectRefResolutionError` — never a silent ``None`` (project no-fallback
rule).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.harness.errors import ObjectRefResolutionError
from molexp.workspace import Workspace
from molexp.workspace.assets.scan import get_asset
from molexp.workspace.folder import Folder

if TYPE_CHECKING:
    from os import PathLike

    from molexp.harness.schemas.change_proposal import ObjectRef

__all__ = ["resolve_object_ref"]


def resolve_object_ref(workspace_root: str | PathLike[str], ref: ObjectRef) -> object:
    """Resolve *ref* to the live workspace entity it names.

    Args:
        workspace_root: The workspace root (callers pass ``ctx.workspace_root``).
        ref: The typed object pointer to resolve.

    Returns:
        The live entity — a ``Run`` (``kind="run"``), ``Experiment``
        (``"experiment"``), ``Folder`` (``"folder"``), or ``Asset``
        (``"data_asset"``).

    Raises:
        ObjectRefResolutionError: The id resolves to nothing, or ``ref.kind`` is
            not one of the supported kinds.
    """
    ws = Workspace(root=workspace_root)
    kind = ref.kind

    if kind == "run":
        for project in ws.list_projects():
            for experiment in project.list_experiments():
                if experiment.has_run(ref.id):
                    return experiment.get_run(ref.id)
        raise ObjectRefResolutionError(f"no run with id {ref.id!r} in workspace {workspace_root}")

    if kind == "experiment":
        for project in ws.list_projects():
            if project.has_experiment(ref.id):
                return project.get_experiment(ref.id)
        raise ObjectRefResolutionError(
            f"no experiment with id {ref.id!r} in workspace {workspace_root}"
        )

    if kind == "folder":
        if not ws.has_folder(ref.id, cls=Folder):
            raise ObjectRefResolutionError(
                f"no folder named {ref.id!r} at workspace root {workspace_root}"
            )
        return ws.get_folder(ref.id, cls=Folder)

    if kind == "data_asset":
        asset = get_asset(workspace_root, ref.id)
        if asset is None:
            raise ObjectRefResolutionError(
                f"no data asset with id {ref.id!r} in workspace {workspace_root}"
            )
        return asset

    raise ObjectRefResolutionError(f"unknown ObjectRef kind {kind!r} (ref id {ref.id!r})")
