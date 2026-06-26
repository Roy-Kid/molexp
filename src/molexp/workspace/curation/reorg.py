"""Reorganize a workspace tree.

Thin compositions over the workspace move / import / delete primitives:

* ``move_run`` relocates a Run to another Experiment (``Run.move_to``).
* ``rehome_asset`` re-imports a ``DataAsset``'s payload into another scope
  (``DataAssetLibrary.import_asset``), preserving its content hash.
* ``delete_folder`` removes a folder and prunes it from its parent's listing.

``reslug`` (entity rename) is intentionally absent: renaming an entity's id must
rewrite its authoritative metadata file and re-home every asset cataloged under
the old scope (and a Run's id is embedded in its execution ids), which is a
focused follow-up rather than a ``move_to`` compose.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from molexp.workspace.assets.data import DataAsset, DataAssetLibrary, ImportAction
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.folder import Folder
    from molexp.workspace.run import Run

__all__ = ["delete_folder", "move_run", "rehome_asset"]


class _ImportTarget(Protocol):
    """A workspace scope that can import a ``DataAsset`` (Workspace / Project /
    Experiment)."""

    @property
    def data_assets(self) -> DataAssetLibrary: ...


def move_run(run: Run, target_experiment: Experiment) -> None:
    """Relocate *run* to *target_experiment*.

    Composes ``Run.move_to``, which is container- and children-index-aware.
    Local-filesystem only — ``move_to`` raises ``NotImplementedError`` on a
    remote-backed folder.

    Args:
        run: The run to move.
        target_experiment: The experiment to move it under.
    """
    run.move_to(target_experiment)


def _scope_dir(entity: object) -> Path:
    """Resolve a workspace entity's on-disk scope directory.

    Probes the entity's directory property in most-specific order so a Run's
    ``run_dir`` wins over a Workspace's ``root``.
    """
    for attr in ("run_dir", "experiment_dir", "project_dir", "root"):
        value = getattr(entity, attr, None)
        if value is not None:
            return Path(str(value))
    raise TypeError(f"{type(entity).__name__} has no resolvable scope directory")


def rehome_asset(
    asset: DataAsset,
    *,
    source: object,
    target: _ImportTarget,
    action: ImportAction = "copy",
) -> DataAsset:
    """Re-import *asset*'s payload into *target*'s scope.

    Composes ``DataAssetLibrary.import_asset`` on the target. For ``action`` of
    ``"copy"`` or ``"move"`` the content hash is recomputed on the destination
    payload, so identical bytes yield an identical ``content_hash``.

    Args:
        asset: The data asset whose payload to re-home.
        source: The workspace entity the asset currently lives under (used only
            to resolve the source payload path).
        target: The destination scope (Workspace / Project / Experiment).
        action: Transfer mode passed through to ``import_asset``.

    Returns:
        The newly imported :class:`DataAsset` under *target*'s scope.
    """
    payload = asset.payload(_scope_dir(source))
    return target.data_assets.import_asset(asset.name, payload, action=action)


def delete_folder(folder: Folder) -> None:
    """Delete *folder* and drop it from its parent's listing.

    Routes through ``parent.remove_folder`` (which also prunes the derived
    children index) when the folder is mounted; falls back to ``Folder.delete``
    for an unmounted folder.

    Args:
        folder: The folder to delete.
    """
    parent = folder.parent
    if parent is not None:
        parent.remove_folder(folder.name, cls=type(folder))
    else:
        folder.delete()
