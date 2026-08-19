"""``Stage`` — ABC for one unit of work in a harness pipeline.

Subclasses pin :attr:`name` (a stable identifier used in events) and
implement :meth:`run` as ``async def run(self, ctx) -> PlanArtifactRef``. Unlike
an agent turn (a coroutine that streams ``AgentEvent``
to an injected sink), a harness ``Stage`` returns a single ``PlanArtifactRef``;
the framing events (``stage_started`` / ``artifact_created`` /
``stage_completed`` / ``stage_failed``) are written by ``StageRunner``, not
by the stage itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from molexp.harness.schemas import PlanArtifactRef

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext

__all__ = ["Stage"]


class Stage(ABC):
    """One unit of work in a harness pipeline."""

    name: ClassVar[str]

    @abstractmethod
    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        """Execute this stage and return its produced artifact.

        Args:
            ctx: services container (artifact store, event log, lineage
                store) bound to the current run.

        Returns:
            The single :class:`PlanArtifactRef` produced by this stage.
            ``StageRunner`` consumes ``ref.parent_ids`` to wire
            ``derived_from`` lineage edges automatically.
        """
        raise NotImplementedError
