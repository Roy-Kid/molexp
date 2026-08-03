"""Unified Target family — one type system for every execution destination.

:class:`~molexp.workspace.models.ComputeTarget` (``models.py``) is the
canonical, persisted registry record (transport x scheduler axes).  This
module contributes the *address views* — :class:`LocalTarget` and
:class:`RemoteTarget` are thin ``ComputeTarget`` subclasses produced by
parsing a CLI target spec, so every target in the system shares one field
surface (``host`` / ``port`` / ``identity_file`` / ``ssh_opts`` /
``scheduler`` / ``scratch_root``).  ``Target`` is the type alias for the
whole family.

Spec notation accepted by :func:`parse_target` / :func:`resolve_target`:

* SCP-style remote: ``user@host:/path`` or ``host:/path``
* Registered compute targets: ``@target-name`` (looked up in the workspace
  registry via :func:`molexp.workspace.targets.resolve_compute_target`,
  which includes the built-in ``local`` fallback)
* Local paths: ``/abs/path``, ``./rel``, or empty (cwd)

Session management caches Transport instances so repeated commands reuse
the same SSH connection.  ``-i/--interactive`` mode builds on this.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molq.transport import Transport

from .fs import FileSystem
from .fs_local import LocalFileSystem
from .models import ComputeTarget
from .targets import resolve_compute_target, to_transport

if TYPE_CHECKING:
    from molexp.workspace.workspace import Workspace


# ---------------------------------------------------------------------------
# Target types — address views over the canonical ComputeTarget
# ---------------------------------------------------------------------------


class LocalTarget(ComputeTarget):
    """A workspace on the local filesystem (address view of ``ComputeTarget``).

    ``host`` is always ``None`` (LocalTransport); the parsed path is stored
    in the canonical ``scratch_root`` slot and surfaced as a :class:`Path`
    via :attr:`path`.
    """

    def __init__(self, path: Path | str | None = None, **data: Any) -> None:  # noqa: ANN401 — forwarded pydantic field values
        if path is not None:
            data.setdefault("scratch_root", str(path))
        data.setdefault("name", data.get("scratch_root", "local"))
        super().__init__(**data)

    @property
    def path(self) -> Path:
        """The workspace directory on this machine."""
        return Path(self.scratch_root)

    def __str__(self) -> str:
        return str(self.path)


class RemoteTarget(ComputeTarget):
    """A workspace accessible via SSH transport (address view).

    Adds the split-out ``user`` slot on top of the canonical fields; the
    remote path is stored in ``scratch_root`` and surfaced as :attr:`path`.
    """

    user: str | None = None

    def __init__(self, *, path: str | None = None, **data: Any) -> None:  # noqa: ANN401 — forwarded pydantic field values
        if path is not None:
            data.setdefault("scratch_root", path)
        if "name" not in data:
            user_part = f"{data['user']}@" if data.get("user") else ""
            data["name"] = f"{user_part}{data.get('host', '')}:{data.get('scratch_root', '')}"
        super().__init__(**data)

    @property
    def path(self) -> str:
        """The workspace directory on the remote host."""
        return self.scratch_root

    @property
    def scp_notation(self) -> str:
        """Return ``user@host:/path`` representation."""
        user_part = f"{self.user}@" if self.user else ""
        return f"{user_part}{self.host}:{self.path}"

    def __str__(self) -> str:
        return self.scp_notation


#: The unified target family. Every target IS a ``ComputeTarget`` (the
#: persisted shape), but resolution only ever produces one of the two address
#: views — so the alias names the union. Aliasing the base instead made
#: ``isinstance(t, RemoteTarget)`` narrow to a type with no ``path`` at all,
#: which is how the CLI ended up unable to name the path it had just resolved.
Target = LocalTarget | RemoteTarget


# SCP pattern: [user@]host:path  (but not :// which is a URL)
_SCP_RE = re.compile(r"^(?:([a-zA-Z0-9_.-]+)@)?([a-zA-Z0-9_.-]+):(.+)$")


def parse_target(raw: str | None) -> Target:
    """Parse a target string into a :class:`Target`.

    Resolution order:
    1. ``None`` or empty → current directory (local)
    2. ``@name`` → not resolved here — raises :class:`TargetNeedsResolution`;
       call :func:`resolve_target` with a workspace to look up the named
       compute target.
    3. ``[user@]host:path`` (SCP-style) → :class:`RemoteTarget`
    4. Everything else → :class:`LocalTarget`
    """
    if not raw:
        return LocalTarget(Path.cwd())

    if raw.startswith("@"):
        # Caller must resolve via the compute-target registry.
        raise TargetNeedsResolution(raw)

    m = _SCP_RE.match(raw)
    if m:
        user, host, path = m.groups()
        port, identity_file = _resolve_ssh_details(host)
        path = _restore_tilde(path)
        return RemoteTarget(
            user=user or None,
            host=host,
            port=port,
            path=path,
            identity_file=identity_file,
        )

    return LocalTarget(Path(raw).expanduser().resolve())


def _restore_tilde(path: str) -> str:
    """Convert shell-expanded local home path back to ``~``.

    In zsh (and bash with certain options), ``~`` after ``:`` is
    expanded by the shell before Python sees it — ``host:~/dir``
    arrives as ``host:/Users/name/dir``.  We convert it back so
    the remote shell expands ``~`` to the remote home directory.
    """
    local_home = os.path.expanduser("~")  # noqa: PTH111
    if path.startswith(local_home + "/"):
        return "~/" + path[len(local_home) + 1 :]
    if path == local_home:
        return "~"
    return path


class TargetNeedsResolution(Exception):
    """Raised by :func:`parse_target` when the input is ``@name`` and needs
    lookup against the workspace's compute-target registry."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Target {name!r} requires resolution against workspace registry")
        self.name = name


def _split_user_host(host_str: str) -> tuple[str | None, str]:
    """Split ``user@host`` into ``(user, hostname)``."""
    if "@" in host_str:
        user, hostname = host_str.rsplit("@", 1)
        return user, hostname
    return None, host_str


def _resolve_ssh_details(host: str) -> tuple[int | None, str | None]:
    """Resolve *host* as an SSH alias, returning ``(port, identity_file)``.

    If the host is a configured SSH alias, its port and identity file are
    extracted so they can be passed explicitly to ``SshTransport``.  The
    host string itself is returned unchanged — ``ssh <alias>`` resolves
    everything else (hostname, ProxyJump, etc.) at connect time.

    When resolution fails (no ``ssh`` binary, unparseable output), returns
    ``(None, None)`` — the raw host is still usable as-is.
    """
    try:
        from molq.ssh_config import resolve_ssh_host

        resolved = resolve_ssh_host(host)
        return resolved.port, resolved.identity_file
    except OSError:
        return None, None


def resolve_target(raw: str | None, ws: Workspace | None = None) -> tuple[Target, Transport]:
    """Fully resolve a target string into a (Target, Transport) pair.

    The single resolution path shared by every CLI command.  When *ws* is
    provided, ``@name`` targets are looked up in the workspace's
    compute-target registry (with the built-in ``local`` fallback — the same
    rule the server applies).  Otherwise ``@name`` raises
    :class:`TargetNeedsResolution`.
    """
    if raw and raw.startswith("@"):
        name = raw[1:]
        if ws is None:
            raise TargetNeedsResolution(raw)
        try:
            ct = resolve_compute_target(ws, name)
        except KeyError:
            raise TargetNotFound(name) from None
        if ct.is_remote and ct.host:
            user, hostname = _split_user_host(ct.host)
            ssh_port, ssh_identity = _resolve_ssh_details(hostname)
            target: Target = RemoteTarget(
                name=ct.name,
                user=user,
                host=hostname,
                port=ct.port or ssh_port,
                path=ct.scratch_root,
                identity_file=ct.identity_file or ssh_identity,
                ssh_opts=list(ct.ssh_opts),
                scheduler=ct.scheduler,
                default_resources=dict(ct.default_resources),
                default_scheduling=dict(ct.default_scheduling),
            )
        else:
            target = LocalTarget(
                Path(ct.scratch_root),
                name=ct.name,
                scheduler=ct.scheduler,
                default_resources=dict(ct.default_resources),
                default_scheduling=dict(ct.default_scheduling),
            )
        return target, to_transport(ct)

    target = parse_target(raw)
    transport = target_to_transport(target)
    return target, transport


class TargetNotFound(Exception):
    """Raised when a ``@name`` target is not found in the workspace registry."""

    def __init__(self, name: str) -> None:
        super().__init__(f"No compute target named {name!r}")


def target_to_transport(target: Target) -> Transport:
    """Build the molq Transport for *target*.

    Thin delegate to :func:`molexp.workspace.targets.to_transport` — every
    member of the unified family is a ``ComputeTarget``, so one transport
    builder serves both the address views and the registry records.
    """
    return to_transport(target)


def target_to_filesystem(target: Target) -> FileSystem:
    """Build the appropriate :class:`FileSystem` for *target*.

    Local targets get a :class:`LocalFileSystem`; remote targets get a
    :class:`RemoteFileSystem` backed by an SSH transport.
    """
    if not target.is_remote:
        return LocalFileSystem()
    from molq.options import SshTransportOptions
    from molq.transport import SshTransport

    from .fs_remote import RemoteFileSystem

    opts = SshTransportOptions(
        host=target.host or "",
        port=target.port,
        identity_file=target.identity_file,
        ssh_opts=tuple(target.ssh_opts),
    )
    transport = SshTransport(options=opts)
    return RemoteFileSystem(transport)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


@dataclass
class SSHSession:
    """A cached SSH Transport connection to a remote target.

    (Renamed from ``Session`` — that name now belongs exclusively to the
    agent layer's LLM conversation session.)
    """

    name: str
    target: RemoteTarget
    transport: Transport
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used = time.time()

    def close(self) -> None:
        """Release the underlying transport if it supports cleanup."""
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()


class SessionManager:
    """Global registry of active SSH sessions, keyed by SCP notation."""

    _sessions: dict[str, SSHSession] = {}  # noqa: RUF012

    @classmethod
    def get(cls, target: RemoteTarget) -> SSHSession | None:
        """Return an existing session for *target*, or ``None``."""
        key = str(target)
        return cls._sessions.get(key)

    @classmethod
    def get_or_create(cls, target: RemoteTarget) -> SSHSession:
        """Return an existing session or create + cache a new one."""
        key = str(target)
        if key in cls._sessions:
            session = cls._sessions[key]
            session.touch()
            return session
        transport = target_to_transport(target)
        session = SSHSession(name=key, target=target, transport=transport)
        cls._sessions[key] = session
        return session

    @classmethod
    def close(cls, name: str) -> bool:
        """Close and remove the named session. Returns ``False`` if not found."""
        session = cls._sessions.pop(name, None)
        if session is None:
            return False
        session.close()
        return True

    @classmethod
    def close_all(cls) -> int:
        """Close all sessions. Returns count of closed sessions."""
        count = len(cls._sessions)
        for session in cls._sessions.values():
            session.close()
        cls._sessions.clear()
        return count

    @classmethod
    def get_by_name(cls, name: str) -> SSHSession | None:
        """Return a session by its name (SCP notation)."""
        return cls._sessions.get(name)

    @classmethod
    def list_sessions(cls) -> list[SSHSession]:
        """Return all active sessions sorted by last use."""
        return sorted(cls._sessions.values(), key=lambda s: s.last_used, reverse=True)


__all__ = [
    "LocalTarget",
    "RemoteTarget",
    "SSHSession",
    "SessionManager",
    "Target",
    "TargetNeedsResolution",
    "TargetNotFound",
    "parse_target",
    "resolve_target",
    "target_to_transport",
]
