"""``approval/request`` waterfall and tools/pre-execute hang."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from molexp.harness.errors import ApprovalPendingError
from molexp.harness.host.context import Context
from molexp.harness.host.keys import Keys
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
from molexp.harness.store.approval_store import ApprovalStore

__all__ = ["ApprovalPlugin", "tool_approval_request_id"]


def tool_approval_request_id(name: str, side_effects: tuple[str, ...]) -> str:
    """Stable id so a stored grant replays for the same tool + effects."""
    effects = ",".join(sorted(side_effects))
    return f"tool:{name}:{effects}"


class ApprovalPlugin:
    """Store-first ``approval/request``; side-effect tools hang on pre-execute."""

    name = "approval_policy"
    inject: tuple[str, ...] = (Keys.APPROVAL,)

    def apply(self, ctx: Context) -> None:
        """Register the two waterfalls."""
        store = ctx.require(Keys.APPROVAL)
        if not isinstance(store, ApprovalStore):
            raise TypeError(f"{Keys.APPROVAL} is not an ApprovalStore")

        async def on_request(value: object, nxt: Callable[..., Awaitable[object]]) -> object:
            if not isinstance(value, ApprovalRequest):
                raise TypeError("approval/request must receive an ApprovalRequest")
            existing = store.granted_decision_for(value.id)
            if existing is not None:
                return existing
            result = await nxt(value)
            if isinstance(result, ApprovalDecision):
                store.record_decision(result)
                return result
            run_id = str(ctx.get(Keys.RUN_ID) or "host")
            store.record_pending(run_id, value)
            raise ApprovalPendingError([value], run_id)

        async def on_pre_execute(payload: object, nxt: Callable[..., Awaitable[object]]) -> object:
            data: dict[str, object] = {}
            if isinstance(payload, dict):
                data = {str(k): v for k, v in payload.items()}
            name = str(data.get("name", ""))
            effects = _side_effects_for(ctx, name)
            if not effects:
                return await nxt(payload)
            request = ApprovalRequest(
                id=tool_approval_request_id(name, effects),
                intent="overwrite",
                reason=(
                    f"tool {name!r} declares destructive side effects "
                    f"{list(effects)}; approval required before invocation"
                ),
                triggered_by_policy="side_effects_present",
                metadata={"tool": name, "side_effects": list(effects)},
                created_at=datetime.now(tz=UTC),
            )
            decision = await ctx.waterfall("approval/request", request)
            if isinstance(decision, ApprovalDecision) and not decision.granted:
                return {"result": "denied", "request_id": decision.request_id}
            return await nxt(payload)

        ctx.on("approval/request", on_request, mode="waterfall")
        ctx.on("tools/pre-execute", on_pre_execute, mode="waterfall")


def _side_effects_for(ctx: Context, name: str) -> tuple[str, ...]:
    belt = ctx.get(Keys.TOOLS)
    if belt is None:
        return ()
    lookup = getattr(belt, "get", None)
    tool = lookup(name) if callable(lookup) else None
    if tool is None:
        return ()
    raw = getattr(tool, "side_effects", ()) or ()
    return tuple(str(item) for item in raw)
