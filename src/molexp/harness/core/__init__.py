"""Stage execution core for ``molexp.harness``.

Plain-Python runtime pieces:

- :class:`HarnessRunContext` — frozen container handing services to stages.
- :class:`Stage` — ABC with ``async run(ctx) -> ArtifactRef``.
- :func:`run_stage_bracketed` — the audit bracket (``stage_started`` /
  ``artifact_created`` / ``stage_completed`` / ``stage_failed`` events +
  ``derived_from`` lineage edges) every stage execution goes through.
- :class:`StageRunner` — thin single-stage wrapper over the bracket.
- :func:`stage_fingerprint` — code identity for the Mode completion ledger.

The harness ``Stage`` returns one ``ArtifactRef``; the agent layer's loops
stream ``AgentEvent`` instead — the two abstractions stay distinct.
"""

from __future__ import annotations

from molexp.harness.core.fingerprint import stage_fingerprint
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.core.stage_runner import StageRunner, run_stage_bracketed

__all__ = [
    "HarnessRunContext",
    "Stage",
    "StageRunner",
    "run_stage_bracketed",
    "stage_fingerprint",
]
