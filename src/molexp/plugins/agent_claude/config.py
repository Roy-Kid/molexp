"""Configuration dataclasses for the Claude CLI coding-agent plugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubagentDef:
    """Provider-neutral declaration of a sub-agent.

    Forwarded to the Claude CLI as a native sub-agent definition (the
    ``--agents`` JSON map). The primary agent dispatches focused work to a
    sub-agent so e.g. routine GitHub state mutations can run on a small
    fast model while the main implementation stays on a large model.

    Attributes:
        name: Stable identifier the primary agent uses to invoke the
            sub-agent (the key in the ``--agents`` map).
        description: One-line summary surfaced to the primary agent so it
            knows when to delegate.
        prompt: System prompt for the sub-agent (its instructions).
        tier: Tier name from the caller's model map (e.g. ``"fast"``,
            ``"primary"``). Resolved against the caller-supplied
            ``models`` dict at ``ClaudeCliClient`` construction time.
    """

    name: str
    description: str
    prompt: str
    tier: str


@dataclass(frozen=True)
class ClaudeCliConfig:
    """Configuration for one Claude CLI session.

    Attributes:
        command: Path to the ``claude`` binary. Defaults to ``"claude"``
            (resolved via ``$PATH``); set to a literal path to bypass.
        permission_mode: Argument forwarded to ``--permission-mode``.
            ``"bypassPermissions"`` skips interactive approval prompts —
            required for non-interactive orchestrated runs.
        model: Explicit model id passed to ``--model``. Wins over
            ``model_tier`` resolution.
        model_tier: Tier name (key in caller-supplied ``models``) used for
            the *main* agent of the session. Sub-agents pin their own
            tier independently.
        mcp_config: Path to a fully-prepared ``mcp.json`` to forward via
            ``--mcp-config``. If ``None``, no MCP servers are wired in.
            The plugin does **not** auto-inject any servers — callers are
            responsible for assembling the file (e.g. symphony pre-renders
            its github_graphql server entry).
        strict_mcp_config: Forward ``--strict-mcp-config`` to the CLI.
        extra_args: Additional positional CLI arguments appended verbatim
            (e.g. provider-specific flags not yet modelled here).
        turn_timeout_ms: Hard ceiling on a single turn (after the first
            event arrives). Raises :class:`AgentError` with code
            ``"turn_timeout"`` if exceeded.
        read_timeout_ms: Time allowed before the **first** event is read.
            Raises :class:`AgentError` with code ``"response_timeout"`` if
            exceeded.
        stall_timeout_ms: Reserved for orchestrator-level stall detection;
            unused by the client itself.
        strip_anthropic_api_key_env: If ``True`` (default), the plugin
            removes ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``, and
            ``CLAUDE_CODE_OAUTH_TOKEN`` from the spawned subprocess
            environment so the CLI falls back to its OAuth credentials in
            ``~/.claude/``.
    """

    command: str = "claude"
    permission_mode: str = "bypassPermissions"
    model: str | None = None
    model_tier: str = "primary"
    mcp_config: Path | None = None
    strict_mcp_config: bool = False
    extra_args: tuple[str, ...] = ()
    turn_timeout_ms: int = 3600000
    read_timeout_ms: int = 5000
    stall_timeout_ms: int = 300000
    strip_anthropic_api_key_env: bool = True
