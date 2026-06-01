"""Stage execution core for ``molexp.harness``.

Three plain-Python runtime classes:

- :class:`HarnessRunContext` — frozen container handing services to stages.
- :class:`Stage` — ABC with ``async run(ctx) -> ArtifactRef``.
- :class:`StageRunner` — wraps each stage with ``stage_started`` /
  ``artifact_created`` / ``stage_completed`` events and auto-wires
  ``derived_from`` provenance edges from the returned ref's ``parent_ids``.

These are intentionally distinct from :class:`molexp.agent.Stage`
(which is an async-generator yielding ``AgentEvent``). The harness layer's
``Stage`` is the workflow-level abstraction returning one ``ArtifactRef``;
the agent layer's ``Stage`` is the mode-pipeline abstraction streaming
events.
"""

from __future__ import annotations

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.core.stage_runner import StageRunner
from molexp.harness.core.stage_task import StageTask, run_stage_bracketed

__all__ = [
    "HarnessRunContext",
    "Stage",
    "StageRunner",
    "StageTask",
    "run_stage_bracketed",
]
