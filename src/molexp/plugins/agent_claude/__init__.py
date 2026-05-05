"""Claude Code CLI as a coding-agent plugin.

Importing this package exposes :class:`ClaudeCliClient`, which spawns the
``claude`` CLI per turn and translates its ``--output-format stream-json``
event stream into normalized :class:`molexp.plugins.coding_agent.TurnResult`
plus an event callback the orchestrator consumes.

Multi-turn continuation is achieved by reusing one v4 UUID across turns:

* turn 1 uses ``--session-id <uuid>`` to seed the CLI session
* turn 2..N use ``--resume <uuid>`` to continue the same session

Subscription auth (``~/.claude/`` OAuth) is preserved by stripping
``ANTHROPIC_API_KEY`` and friends from the subprocess environment unless the
operator opts out via ``ClaudeCliConfig.strip_anthropic_api_key_env=False``.
"""

from __future__ import annotations

from molexp.plugins.agent_claude.client import ClaudeCliClient
from molexp.plugins.agent_claude.config import ClaudeCliConfig, SubagentDef

__all__ = ["ClaudeCliClient", "ClaudeCliConfig", "SubagentDef"]
