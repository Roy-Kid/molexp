"""Workspace-layer cache folder — content-addressed entries on disk.

``CacheFolder`` is a :class:`~molexp.workspace.folder.Folder` subclass
rooted at ``<workspace_root>/cache/``. It vendors string-keyed
``.json`` entry files and exposes :meth:`CacheFolder.as_cache_store` —
an adapter that satisfies the workflow-layer
:class:`molexp.workflow.cache_store.CacheStore` Protocol without
forcing workflow into ``import molexp.workspace`` at module load.
"""

from molexp.workspace.cache.folder import (
    WORKSPACE_CACHE_KIND,
    CacheFolder,
)

__all__ = [
    "WORKSPACE_CACHE_KIND",
    "CacheFolder",
]
