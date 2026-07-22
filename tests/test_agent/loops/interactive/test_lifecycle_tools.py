"""``lifecycle_tools`` — the opt-in workspace cancel/harvest verbs.

The gating (default operation_mode does not mount these; ``lifecycle`` mode
does) is owned by ``test_loop.py``; the agent→harness firewall is owned by
``tests/test_agent/test_import_guard.py``. This file only locks the factory's
tool surface.
"""

from __future__ import annotations

from pathlib import Path

from molexp.agent.loops.interactive.lifecycle_tools import (
    LIFECYCLE_TOOL_NAMES,
    lifecycle_tools,
)


class TestLifecycleTools:
    def test_expose_cancel_and_harvest_verbs(self, tmp_path: Path) -> None:
        names = {tool.__name__ for tool in lifecycle_tools(workspace_root=tmp_path)}
        assert names == LIFECYCLE_TOOL_NAMES == {"cancel_run", "harvest_run"}
