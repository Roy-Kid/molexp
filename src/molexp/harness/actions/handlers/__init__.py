"""Concrete :class:`ChangeActionHandler`s for guarded execution.

Slice 02 ships the curation handlers (``asset_move`` / ``artifact_delete``) over
:mod:`molexp.workspace.curation`. Later slices register more handlers (run
lifecycle, workflow/param change) onto the same slice-01 registry.
"""

from __future__ import annotations

from molexp.harness.actions.handlers.curation import (
    ArtifactDeleteHandler,
    AssetMoveHandler,
    register_curation_handlers,
)

__all__ = ["ArtifactDeleteHandler", "AssetMoveHandler", "register_curation_handlers"]
