"""``HookRegistry`` dispatch + typed-context tests (spec ac-008)."""

from __future__ import annotations

import pytest

from molexp.agent.harness.hooks import HookContext, HookPoint, HookRegistry


def test_hook_point_has_five_members() -> None:
    members = {p.value for p in HookPoint}
    assert members == {
        "before_stage",
        "after_stage",
        "before_approval",
        "before_compact",
        "before_model_call",
    }


@pytest.mark.asyncio
async def test_handlers_fire_in_registration_order() -> None:
    registry = HookRegistry()
    order: list[int] = []

    async def first(ctx: HookContext) -> None:
        order.append(1)

    async def second(ctx: HookContext) -> None:
        order.append(2)

    async def third(ctx: HookContext) -> None:
        order.append(3)

    registry.register(HookPoint.before_stage, first)
    registry.register(HookPoint.before_stage, second)
    registry.register(HookPoint.before_stage, third)

    await registry.dispatch(HookPoint.before_stage, HookContext(point=HookPoint.before_stage))
    assert order == [1, 2, 3]


@pytest.mark.asyncio
async def test_dispatch_with_no_handlers_is_a_noop() -> None:
    registry = HookRegistry()
    results = await registry.dispatch(
        HookPoint.before_compact, HookContext(point=HookPoint.before_compact)
    )
    assert results == ()


@pytest.mark.asyncio
async def test_typed_return_values_propagate() -> None:
    registry = HookRegistry()

    async def approve(ctx: HookContext) -> dict[str, bool]:
        return {"approved": True}

    async def veto(ctx: HookContext) -> dict[str, bool]:
        return {"approved": False}

    registry.register(HookPoint.before_approval, approve)
    registry.register(HookPoint.before_approval, veto)

    results = await registry.dispatch(
        HookPoint.before_approval, HookContext(point=HookPoint.before_approval)
    )
    assert results == ({"approved": True}, {"approved": False})


@pytest.mark.asyncio
async def test_handler_returning_none_is_dropped_from_results() -> None:
    registry = HookRegistry()

    async def silent(ctx: HookContext) -> None:
        return None

    async def speaks(ctx: HookContext) -> str:
        return "hi"

    registry.register(HookPoint.after_stage, silent)
    registry.register(HookPoint.after_stage, speaks)
    results = await registry.dispatch(
        HookPoint.after_stage, HookContext(point=HookPoint.after_stage)
    )
    assert results == ("hi",)


@pytest.mark.asyncio
async def test_hook_context_carries_payload() -> None:
    registry = HookRegistry()
    seen: list[HookContext] = []

    async def capture(ctx: HookContext) -> None:
        seen.append(ctx)

    registry.register(HookPoint.before_stage, capture)
    ctx = HookContext(point=HookPoint.before_stage, stage_name="draft", payload={"k": 1})
    await registry.dispatch(HookPoint.before_stage, ctx)
    assert seen[0].stage_name == "draft"
    assert seen[0].payload == {"k": 1}


def test_registry_handlers_are_per_point_isolated() -> None:
    registry = HookRegistry()

    async def handler(ctx: HookContext) -> None:
        return None

    registry.register(HookPoint.before_stage, handler)
    assert registry.handlers(HookPoint.before_stage) == (handler,)
    assert registry.handlers(HookPoint.after_stage) == ()
