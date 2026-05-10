"""Platform-default MCP server registry seed.

The molcrafts ecosystem ships an MCP gateway (``molmcp``) that fronts
molpy / molpack / molrs / lammps / molexp behind one MCP transport.
Rather than treating it as a parallel "platform default" branch, we
seed it as an ordinary user-scope entry into ``~/.molexp/mcp.json`` on
first :class:`~molexp.agent.mcp.store.McpStore` construction. Once
seeded, the entry is indistinguishable from any user-added one — the
user can edit, override at workspace scope, or delete it; the sentinel
file (``.mcp_seeded``) records which default names have already been
seeded once so deletions stick across re-runs.

The module is import-cheap: nothing here pulls ``pydantic_ai`` or
``pydantic_graph`` (the import-guard test under
:mod:`tests.test_agent.test_import_guard` enforces this). The
``MOLEXP_MOLMCP_COMMAND`` environment variable is honoured *only at
seed time*; subsequent runs treat the on-disk JSON as the source of
truth.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mollog import get_logger

__all__ = [
    "MCP_DEFAULTS",
    "MCP_SEEDED_FILENAME",
    "MOLMCP_COMMAND_ENV",
    "MOLMCP_USAGE_INSTRUCTIONS",
    "seed_user_defaults",
]


_LOG = get_logger(__name__)


MCP_SEEDED_FILENAME = ".mcp_seeded"
"""Sentinel filename in the User config dir; tracks which default names
have already been seeded once. Disable-by-deletion relies on the
combination of (a) the sentinel listing the name and (b) the name being
absent from ``mcp.json`` — see :func:`seed_user_defaults`."""

MOLMCP_COMMAND_ENV = "MOLEXP_MOLMCP_COMMAND"
"""Environment variable read at *seed time only*. ``shlex.split`` is
used to parse a ``command + args`` line; an empty / whitespace-only
value resolves to the documented default."""


MOLMCP_USAGE_INSTRUCTIONS = (
    "You have access to the molcrafts ecosystem (molpy, molpack, molrs, "
    "lammps, molexp) through `molmcp__*` tools. Before writing fresh "
    "Python or LAMMPS input by hand, inspect the available `molmcp__*` "
    "tools — they expose battle-tested builders, parametrizers, and "
    "inspectors for the same tasks."
)
"""Per-server prompt fragment surfaced to the LLM. The runner
concatenates the strings from every active entry's
``usage_instructions`` and prepends them to the mode's system prompt
— see :class:`~molexp.agent.runner.AgentRunner` for the wiring."""


_MOLMCP_DEFAULT_COMMAND = "molmcp"
_MOLMCP_DEFAULT_ARGS: tuple[str, ...] = ("gateway",)


MCP_DEFAULTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "molmcp",
        {
            "type": "stdio",
            "command": _MOLMCP_DEFAULT_COMMAND,
            "args": list(_MOLMCP_DEFAULT_ARGS),
            "env": {},
            "usage_instructions": MOLMCP_USAGE_INSTRUCTIONS,
        },
    ),
)
"""Platform-default MCP servers seeded into the User config on first
:class:`McpStore` construction. Each tuple is
``(name, mcp.json-shape spec dict)``; the spec dict round-trips
through :class:`~molexp.agent.mcp.store.StdioSpec` /
:class:`~molexp.agent.mcp.store.HttpSpec` exactly like a user-authored
entry."""


# ── Public API ─────────────────────────────────────────────────────────────


def seed_user_defaults(config_path: Path, sentinel_path: Path) -> bool:
    """Seed :data:`MCP_DEFAULTS` into ``config_path``.

    Behavior:

    1. Read the existing ``mcp.json`` (or treat absent as empty).
    2. Read the sentinel listing previously-seeded names (or treat
       absent as empty).
    3. For each ``(name, spec)`` in :data:`MCP_DEFAULTS`: skip if the
       name is already present in the config OR is listed in the
       sentinel (the user has either already received it, or
       explicitly deleted it after a previous seed).
    4. If any new entries land, write the merged config back atomically
       (temp + ``os.replace``) and update the sentinel to include the
       newly-seeded names.
    5. Read-only / unwritable HOMEs are tolerated: the function emits a
       :func:`mollog.get_logger` warning and returns ``False`` rather
       than propagating ``OSError``.

    Args:
        config_path: ``~/.molexp/mcp.json`` (or a test-scope fake).
        sentinel_path: ``~/.molexp/.mcp_seeded`` (or a test-scope fake).

    Returns:
        ``True`` if the file was modified, ``False`` otherwise (already
        seeded, all defaults user-deleted, or write failed).
    """
    try:
        servers = _read_servers(config_path)
        seeded_names = _read_sentinel(sentinel_path)

        added: list[str] = []
        for name, spec in MCP_DEFAULTS:
            if name in servers:
                continue
            if name in seeded_names:
                # Previously seeded; user deleted it. Respect the deletion.
                continue
            servers[name] = _apply_command_override(name, dict(spec))
            added.append(name)

        if not added:
            return False

        _write_servers(config_path, servers)
        seeded_names.update(added)
        _write_sentinel(sentinel_path, seeded_names)
    except OSError as exc:
        _LOG.warning(
            f"[mcp.defaults] could not seed {config_path}: {exc!r}; skipping. "
            "The agent will run without the default MCP entries."
        )
        return False
    else:
        return True


# ── Internals ──────────────────────────────────────────────────────────────


def _apply_command_override(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Honour the ``MOLEXP_MOLMCP_COMMAND`` env var at seed time.

    Only the ``molmcp`` entry currently looks at the env var. Future
    defaults may grow their own override hooks; today this function is a
    no-op for any other name.
    """
    if name != "molmcp":
        return spec
    raw = os.environ.get(MOLMCP_COMMAND_ENV, "")
    if not raw.strip():
        return spec
    tokens = shlex.split(raw)
    if not tokens:
        return spec
    spec["command"] = tokens[0]
    spec["args"] = list(tokens[1:])
    return spec


def _read_servers(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        content = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    raw = content.get("mcpServers") if isinstance(content, dict) else None
    if not isinstance(raw, dict):
        return {}
    return dict(raw)


def _write_servers(config_path: Path, servers: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"mcpServers": servers}, indent=2, ensure_ascii=False))
    os.replace(tmp, config_path)


def _read_sentinel(sentinel_path: Path) -> set[str]:
    if not sentinel_path.exists():
        return set()
    try:
        content = json.loads(sentinel_path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    seeded = content.get("seeded") if isinstance(content, dict) else None
    if not isinstance(seeded, list):
        return set()
    return {str(name) for name in seeded if isinstance(name, str)}


def _write_sentinel(sentinel_path: Path, seeded_names: Iterable[str]) -> None:
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sentinel_path.with_suffix(".tmp")
    payload = {"seeded": sorted(set(seeded_names))}
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, sentinel_path)
