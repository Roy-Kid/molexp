"""Invariant lock: forbid resurrection of parallel-to-pydantic-ai subpackages.

``agent-pydanticai-rectification`` deleted the dead subpackages that
re-implemented what pydantic-ai already provides (tools / context / memory /
recovery / skills, plus three ``mcp/`` modules). This permanent guard fails
if any of them reappears on disk.

The check is filesystem-only and imports nothing from :mod:`molexp`, so it
collects even when an unrelated downstream import is broken.
"""

from __future__ import annotations

from pathlib import Path

# tests/test_agent/<this>.py → parents[2] is repo root.
_AGENT_SRC = Path(__file__).resolve().parents[2] / "src" / "molexp" / "agent"

# Every path the rectification deleted. Directories have no trailing slash;
# file paths use POSIX separators so the literal works on every OS.
DEAD_PATHS: tuple[str, ...] = (
    "tools",
    "context",
    "memory",
    "recovery",
    "skills",
    "mcp/source.py",
    "mcp/tool_store.py",
    "mcp/probe.py",
)


def test_dead_parallel_subpackages_absent() -> None:
    """None of the deleted parallel-to-pydantic-ai paths may exist."""
    alive = [rel for rel in DEAD_PATHS if (_AGENT_SRC / rel).exists()]
    assert not alive, (
        "agent/ contains parallel-to-pydantic-ai dead code that "
        f"agent-pydanticai-rectification deleted: {alive}"
    )
