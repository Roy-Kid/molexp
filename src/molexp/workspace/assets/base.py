"""Abstract Asset base, Producer, and AssetScope.

The concrete subclasses live in sibling modules.  The discriminated
union ``AnyAsset`` and its ``TypeAdapter`` are built in ``_adapter.py``
to avoid circular imports.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Producer(BaseModel):
    """Who produced this asset.

    At least one field is set for run-produced assets; ``None`` for
    user-imported ``DataAsset``.

    ``inputs`` is the list of upstream ``asset_id``s that were consumed
    to produce this asset — together they form the asset-level lineage
    DAG that :func:`molexp.workspace.assets.lineage.ancestors` and
    :func:`~molexp.workspace.assets.lineage.descendants` traverse.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str | None = None
    execution_id: str | None = None
    task_id: str | None = None
    inputs: tuple[str, ...] = ()


class AssetScope(BaseModel):
    """Identifies which scope (workspace/project/experiment/run) owns an asset.

    ``ids`` is the chain of parent IDs ending with the leaf scope's own id.
    Empty for workspace scope.

    Examples::

        AssetScope(kind="workspace", ids=())
        AssetScope(kind="project", ids=("qm9",))
        AssetScope(kind="experiment", ids=("qm9", "baseline"))
        AssetScope(kind="run", ids=("qm9", "baseline", "run-abc"))
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["workspace", "project", "experiment", "run"]
    ids: tuple[str, ...] = ()

    @property
    def urn(self) -> str:
        """URN fragment: ``workspace`` or ``run/proj/exp/run-id``."""
        if not self.ids:
            return self.kind
        return f"{self.kind}/{'/'.join(self.ids)}"

    @property
    def scope_id(self) -> str:
        """Flat identifier used as catalog key.

        For workspace: the constant ``workspace``.  Otherwise the last
        segment of ``ids`` (the leaf scope's own id).
        """
        return self.ids[-1] if self.ids else "workspace"


class Asset(BaseModel):
    """Shared asset fields.

    Concrete kinds add their own fields and methods; each declares
    ``kind: Literal[...] = "..."`` for the discriminated union.

    ``path`` is relative to the scope's on-disk directory — crucial
    for portability.  Callers resolve it with ``absolute_path``.
    """

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    name: str
    scope: AssetScope
    path: Path
    created_at: datetime
    updated_at: datetime
    producer: Producer | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    content_hash: str | None = None
    """SHA-256 of the asset's payload, prefixed with ``sha256:``.

    Set automatically by :class:`~molexp.workspace.assets.data.DataAssetLibrary`
    and :class:`~molexp.workspace.assets.accessors.ArtifactAccessor`. Streaming
    or time-series kinds (``log``, ``checkpoint``, ``error_trace``,
    ``execution_state``) leave it ``None`` — content addressing of an
    open-ended stream is ambiguous.
    """

    @property
    def uri(self) -> str:
        """Stable URI: ``asset://{scope_urn}/{asset_id}``."""
        return f"asset://{self.scope.urn}/{self.asset_id}"

    def absolute_path(self, scope_dir: Path) -> Path:
        """Resolve ``path`` against the scope's on-disk directory."""
        return scope_dir / self.path
