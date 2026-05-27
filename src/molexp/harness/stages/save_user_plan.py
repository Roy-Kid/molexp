"""``SaveUserPlan`` — first stage of the §3 pipeline.

Writes the user's verbatim natural-language input as a ``user_plan``-kind
artifact (the audit anchor) and then wraps it in a structured
:class:`UserPlan` JSON (also kind ``user_plan``) whose ``parent_ids``
references the raw artifact. The structured ref is what every downstream
stage depends on; :class:`StageRunner` wires the raw → structured
``derived_from`` edge automatically from ``parent_ids``.

Two artifacts under the same kind is intentional: the JSON is a canonical
envelope around the text, not an additional information source. Trying to
collapse them into one would either:

- drop the verbatim text (audit loss), or
- conflate "what the user wrote" with "the harness's structured view".
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.schemas import ArtifactRef, UserPlan

__all__ = ["SaveUserPlan"]


class SaveUserPlan(Stage):
    """Persist the user's plan as raw text + structured JSON."""

    name: ClassVar[str] = "save_user_plan"

    def __init__(
        self,
        user_text: str,
        *,
        user_id: str | None = None,
    ) -> None:
        self._user_text = user_text
        self._user_id = user_id

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        raw_ref = ctx.artifact_store.put_text(
            kind="user_plan",
            text=self._user_text,
            created_by="user",
            parent_ids=[],
        )
        plan = UserPlan(
            raw_text=self._user_text,
            user_id=self._user_id,
            submitted_at=datetime.now(tz=UTC),
        )
        return ctx.artifact_store.put_json(
            kind="user_plan",
            obj=plan.model_dump(mode="json"),
            created_by="SaveUserPlan",
            parent_ids=[raw_ref.id],
        )
