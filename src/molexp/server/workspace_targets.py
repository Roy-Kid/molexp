"""Server-process WorkspaceTarget registry — descriptors for remote
workspace roots.

A :class:`WorkspaceTarget` describes *which remote root to open* — it
exists *before* any workspace is open, so it cannot live on a workspace.
The registry is stored at ``~/.molexp/workspace_targets.json`` and
persisted via :func:`molexp.workspace.base.atomic_write_json` so a crash
mid-write leaves the previous file intact.

This module sits in ``molexp.server`` (Layer 4) and intentionally does
not import ``molexp.agent`` or ``molexp.workflow``.  It reaches downward
into ``molexp.workspace`` for two pieces: ``atomic_write_json`` (the
cross-layer atomic-I/O primitive) and the ``FileSystem`` /
``RemoteFileSystem`` types used by
:func:`target_to_filesystem_for_workspace_target`.
"""

from __future__ import annotations

import builtins
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from molexp.workspace.base import atomic_write_json
from molexp.workspace.fs import FileSystem

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


__all__ = [
    "WorkspaceTarget",
    "WorkspaceTargetRegistry",
    "default_workspace_targets_path",
    "target_to_filesystem_for_workspace_target",
]


# Slug-shaped name: ASCII letters / digits / dot / dash / underscore, ≥ 1 char.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class WorkspaceTarget(BaseModel):
    """Frozen descriptor for a remote workspace root.

    The descriptor is purely declarative — no live connection is held by
    a :class:`WorkspaceTarget` instance.  Conversion to a usable
    :class:`~molexp.workspace.fs.FileSystem` happens at the route
    boundary through :func:`target_to_filesystem_for_workspace_target`.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=False)

    name: str = Field(..., description="Unique slug-shaped identifier")
    host: str = Field(..., description="``user@host`` or bare hostname for SSH")
    port: int | None = Field(default=None, description="SSH port (default ssh config)")
    identity_file: str | None = Field(
        default=None,
        description="Absolute path to an SSH identity file",
    )
    ssh_opts: tuple[str, ...] = Field(
        default=(),
        description="Extra ``ssh`` argv tokens (kept as a tuple so the model stays hashable)",
    )
    root_path: str = Field(
        ...,
        description=(
            "Absolute POSIX path on the remote host that is the workspace root. "
            "Not converted to :class:`pathlib.Path` because local resolution would "
            "rewrite tildes / case to the local host's conventions."
        ),
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _NAME_RE.fullmatch(v):
            raise ValueError(
                f"WorkspaceTarget.name must be a slug (alphanumerics + . _ -); got {v!r}"
            )
        return v

    @field_validator("ssh_opts", mode="before")
    @classmethod
    def _coerce_ssh_opts(cls, v: object) -> tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, (list, tuple)):
            return tuple(str(item) for item in v)
        raise TypeError("ssh_opts must be a sequence of strings")


def default_workspace_targets_path() -> Path:
    """Default on-disk location of the registry — ``~/.molexp/workspace_targets.json``."""
    return Path.home() / ".molexp" / "workspace_targets.json"


# JSON envelope version — bumps if the on-disk shape ever changes.
_STORE_FORMAT_VERSION = 1


class WorkspaceTargetRegistry:
    """Server-process descriptor registry.

    Single-writer assumed (one server process); concurrent server
    processes would race on the JSON file.  In practice ``molexp serve``
    is single-process; the assumption holds.
    """

    def __init__(self, store_path: Path | None = None) -> None:
        self._store_path = store_path or default_workspace_targets_path()
        self._cache: dict[str, WorkspaceTarget] | None = None  # lazy load

    @property
    def store_path(self) -> Path:
        return self._store_path

    # ── public surface ────────────────────────────────────────────────

    def list(self) -> builtins.list[WorkspaceTarget]:
        """Return a fresh snapshot in insertion order."""
        return [*self._load().values()]

    def get(self, name: str) -> WorkspaceTarget:
        """Return the named target or raise :class:`KeyError`."""
        cache = self._load()
        if name not in cache:
            raise KeyError(name)
        return cache[name]

    def has(self, name: str) -> bool:
        return name in self._load()

    def add(self, target: WorkspaceTarget) -> WorkspaceTarget:
        """Register a new target; raises :class:`ValueError` if the name is taken."""
        cache = self._load()
        if target.name in cache:
            raise ValueError(f"workspace target {target.name!r} already exists")
        snapshot = dict(cache)
        snapshot[target.name] = target
        self._persist(snapshot)
        # Only mutate the in-memory cache *after* the disk write succeeds.
        self._cache = snapshot
        return target

    def remove(self, name: str) -> None:
        """Remove the named target; raises :class:`KeyError` if absent."""
        cache = self._load()
        if name not in cache:
            raise KeyError(name)
        snapshot = dict(cache)
        del snapshot[name]
        self._persist(snapshot)
        self._cache = snapshot

    # ── internals ─────────────────────────────────────────────────────

    def _load(self) -> dict[str, WorkspaceTarget]:
        if self._cache is not None:
            return self._cache
        if not self._store_path.exists():
            self._cache = {}
            return self._cache
        try:
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"corrupt registry at {self._store_path}: {exc}") from exc
        try:
            entries = raw["targets"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"corrupt registry at {self._store_path}: missing 'targets' list"
            ) from exc

        loaded: dict[str, WorkspaceTarget] = {}
        for entry in entries:
            target = WorkspaceTarget.model_validate(entry)
            loaded[target.name] = target
        self._cache = loaded
        return self._cache

    def _persist(self, snapshot: dict[str, WorkspaceTarget]) -> None:
        payload = {
            "version": _STORE_FORMAT_VERSION,
            "targets": [t.model_dump(mode="json") for t in snapshot.values()],
        }
        atomic_write_json(self._store_path, payload)


def target_to_filesystem_for_workspace_target(target: WorkspaceTarget) -> FileSystem:
    """Build a :class:`RemoteFileSystem` for a :class:`WorkspaceTarget`.

    Parallels :func:`molexp.workspace.target.target_to_filesystem` but
    works on the server-process descriptor type rather than the
    workspace-scoped ``Target`` sum.  Importing :class:`SshTransport` /
    :class:`SshTransportOptions` is deferred to keep
    ``molexp.server.workspace_targets`` cheap to import.
    """
    from molq.options import SshTransportOptions
    from molq.transport import SshTransport

    from molexp.workspace.fs_remote import RemoteFileSystem

    options = SshTransportOptions(
        host=target.host,
        port=target.port,
        identity_file=target.identity_file,
        ssh_opts=tuple(target.ssh_opts),
    )
    transport = SshTransport(options=options)
    return RemoteFileSystem(transport)
