"""Sentinel: forbid resurrection of parallel-to-pydantic-ai subpackages.

Phase 0 of ``agent-pydanticai-rectification`` introduces this test with
``xfail(strict=True)``: the dead subpackages still exist on disk at Phase 0,
so the assertion below intentionally fails (xfail). Phase 1 runs ``git rm``
on every listed path; the assertion then passes, which surfaces as XPASS and
fails the strict-xfail run on purpose. The maintainer must remove the
``xfail`` marker as the last step of Phase 1 — at that point the test becomes
a permanent guard against re-introduction.

The check is filesystem-only and imports nothing from :mod:`molexp` so it
collects even when an unrelated downstream import (e.g.
``molq.options.SshTransportOptions``) is broken.
"""

from __future__ import annotations

from pathlib import Path

# tests/test_agent/<this>.py → parents[2] is repo root.
_AGENT_SRC = Path(__file__).resolve().parents[2] / "src" / "molexp" / "agent"

# Every relative path that Phase 1 of agent-pydanticai-rectification deletes.
# Directories must end without trailing slash; individual file paths use POSIX
# separators so the literal works on every OS.
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
    """After Phase 1, none of the listed dead paths may exist."""
    alive = [rel for rel in DEAD_PATHS if (_AGENT_SRC / rel).exists()]
    assert not alive, (
        "agent/ contains parallel-to-pydantic-ai dead code that Phase 1 of "
        f"agent-pydanticai-rectification must have deleted: {alive}"
    )
