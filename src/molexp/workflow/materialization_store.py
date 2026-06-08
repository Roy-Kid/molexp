"""Engine-side materialization for the pure-task-context contract.

When a task is a referentially-transparent ``execute(inputs, config) -> output``
function, two responsibilities that used to live *inside* task bodies (reaching
through ``run_context``) move to the engine:

1. **Content-addressed workdir** — :meth:`MaterializationStore.workdir_for` maps
   a node's content identity (the ``code+config+inputs`` hash pinned by
   ``pure-task-context-01-cache-contract``) to a deterministic directory. Same
   identity ⇒ same directory, so cross-run reuse falls out of content
   addressing instead of a hand-rolled namespace key.
2. **Return-value persistence** — :meth:`MaterializationStore.persist_result`
   persists a task's return value as a content-hashed artifact and registers its
   lineage, so a task only has to ``return`` its product (no ``artifact.save``).

``Caching`` (``workflow.cache``) owns the cache *key*; this store owns where a
node *works* and where its *output* lands. Like ``cache_store.py`` it is a
``Protocol`` + a ``File`` implementation, routed through the workspace's own
artifact accessor — it never imports ``molexp.harness`` (the layer DAG forbids
``workflow → harness``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from mollog import get_logger

from .protocols import RunContextLike, TaskOutput

logger = get_logger(__name__)


def _is_json_safe(value: object) -> bool:
    """Return True iff *value* round-trips through ``json.dumps`` cleanly."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


@runtime_checkable
class MaterializationStore(Protocol):
    """Engine-owned workdir derivation + task-return persistence.

    Implementations decide *where* a node works (content-addressed) and *how* a
    return value becomes a durable artifact; the engine calls them so task
    bodies stay pure ``execute(inputs, config) -> output`` functions.
    """

    def workdir_for(self, content_id: str) -> Path:
        """Return the content-addressed working directory for *content_id*.

        Deterministic: the same ``content_id`` always maps to the same path,
        and distinct ids map to distinct paths.
        """

    def persist_result(
        self,
        name: str,
        result: TaskOutput,
        *,
        run_context: RunContextLike | None,
    ) -> str | None:
        """Persist task *name*'s *result* as an artifact; return its content hash.

        Returns the ``"sha256:…"`` content hash on success, or ``None`` when the
        result is not JSON-serializable or no run context is reachable
        (fail-soft — the body's result is unaffected, persistence is skipped).
        """


class FileMaterializationStore:
    """Filesystem :class:`MaterializationStore` rooted at ``store_dir``.

    ``workdir_for`` carves a stable subdirectory from the content id (the hex
    body of a ``"sha256:…"`` / ``"code:config"`` key). ``persist_result``
    delegates byte-writing, content hashing, and lineage registration to the
    run's own ``artifact`` accessor (``run_context.artifact.save``) — the same
    path a workspace-backed run already uses — so this store adds no second
    persistence mechanism.
    """

    def __init__(self, store_dir: Path | str) -> None:
        self._store_dir = Path(store_dir)

    @property
    def store_dir(self) -> Path:
        return self._store_dir

    @staticmethod
    def _slug(content_id: str) -> str:
        """Stable filesystem slug from a content id.

        Strips a leading ``sha256:`` algorithm prefix, then sanitizes the WHOLE
        remaining id (so a ``code:config`` key keeps both terms — splitting on
        ``:`` would drop the code term and collide distinct nodes).
        """
        body = content_id[len("sha256:") :] if content_id.startswith("sha256:") else content_id
        safe = "".join(c if c.isalnum() else "_" for c in body)
        return safe or "root"

    def workdir_for(self, content_id: str) -> Path:
        workdir = self._store_dir / self._slug(content_id)
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    def persist_result(
        self,
        name: str,
        result: TaskOutput,
        *,
        run_context: RunContextLike | None,
    ) -> str | None:
        if run_context is None:
            return None
        if not _is_json_safe(result):
            logger.debug(f"materialize: result for task {name!r} not JSON-safe, skipped")
            return None
        artifact_accessor = getattr(run_context, "artifact", None)
        save = getattr(artifact_accessor, "save", None)
        if not callable(save):
            return None
        try:
            asset = save(name, result)
        except Exception:
            logger.debug(f"materialize: persist of task {name!r} skipped")
            return None
        return getattr(asset, "content_hash", None)


__all__ = [
    "FileMaterializationStore",
    "MaterializationStore",
]
