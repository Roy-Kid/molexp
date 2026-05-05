"""Configuration dataclass for the Codex app-server coding-agent plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CodexConfig:
    """Configuration for one Codex app-server session.

    Attributes:
        command: Shell command line that launches the app-server. Run
            via ``bash -lc <command>`` so users can put ``cd`` / env-var
            preludes in front. Default ``"codex app-server"`` assumes a
            properly installed ``codex`` binary.
        approval_policy: Forwarded to ``thread/start`` and ``turn/start``.
            ``None`` falls back to the app-server's default.
        thread_sandbox: Forwarded to ``thread/start`` as ``sandbox``.
        turn_sandbox_policy: Forwarded to ``turn/start`` as ``sandboxPolicy``.
        turn_timeout_ms: Hard ceiling on a single turn waiting for
            ``turn/completed``. Raises :class:`AgentError` with code
            ``"turn_timeout"`` on timeout.
        read_timeout_ms: Per-request timeout for replies to JSON-RPC
            requests (``initialize``, ``thread/start``, ``turn/start``).
        stall_timeout_ms: Reserved for orchestrator-level stall detection.
        model: Forwarded to ``thread/start`` and ``turn/start`` as
            ``model``. ``None`` lets the app-server pick the default.
    """

    command: str = "codex app-server"
    approval_policy: Any = None
    thread_sandbox: Any = None
    turn_sandbox_policy: Any = None
    turn_timeout_ms: int = 3600000
    read_timeout_ms: int = 5000
    stall_timeout_ms: int = 300000
    model: str | None = None
