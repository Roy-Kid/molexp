"""Service keys published on :class:`~molexp.harness.host.context.Context`.

These are the ``ctx.<key>`` names from ``.claude/notes/harness-plugins.md``.
Consumers inject the key; they do not import a concrete provider package
at the composition boundary.
"""

from __future__ import annotations

__all__ = ["Keys"]


class Keys:
    """``ctx.<name>`` catalog from ``harness-plugins.md`` §2.1.

    Spine names exist even before a provider mounts so ``hasattr(ctx, name)``
    and ``Keys.X`` stay stable. Do not add names that are not in that table.
    Do not add ``AGENT_LOOP``.
    """

    LLM = "llm"
    TOOLS = "tools"
    FS = "fs"
    APPROVAL = "approval"
    SESSIONS = "sessions"
    SYSTEM_PROMPT = "systemPrompt"
    JOBS = "jobs"
    SANDBOX = "sandbox"
    COMMANDS = "commands"
    SETTINGS = "settings"
    CREDENTIALS = "credentials"
    WORKSPACE = "workspace"
    WORKFLOW = "workflow"
    ARTIFACTS = "artifacts"
    EVENTS = "events"
    LINEAGE = "lineage"
    CAPABILITIES = "capabilities"
    EXECUTOR = "executor"
    RUN_ID = "run_id"
    WORKSPACE_ROOT = "workspace_root"
