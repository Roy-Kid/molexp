"""Source-grounded debug-loop repair agent — offline tests.

`_build_repair_agent` wires a `pydantic_ai.Agent[None, RepairDecision]`
with `toolsets=[MCPToolset(...)]` (or injectable `tools=` for tests).
`build_repair_callable` returns a closure that drives it once per call,
or `None` when no molmcp `stdio` entry is configured.

These tests use `FunctionModel` + fake source-introspection callables —
no live LLM, no real MCP subprocess.
"""

from __future__ import annotations

import asyncio

import pytest

from molexp.agent._pydanticai.debug_repair import (
    _build_repair_agent,
    build_repair_callable,
)
from molexp.agent.modes.author.codegen import RepairDecision, TaskImplDraft

pytest.importorskip("pydantic_ai")


def test_build_repair_agent_returns_pydantic_ai_agent() -> None:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    assert isinstance(_build_repair_agent(TestModel()), Agent)


@pytest.mark.asyncio
async def test_repair_agent_invokes_find_capability_on_attribute_error() -> None:
    """The MCP-attached repair agent invokes `molmcp_find_capability` to
    locate the real API for the missing symbol before drafting the impl —
    browse-then-select, same protocol as the drafter."""
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    find_calls: list[str] = []

    async def molmcp_find_capability(task: str) -> dict:
        """Fake molmcp_find_capability — finds the real OPLS-AA typifier."""
        find_calls.append(task)
        return {
            "matches": [
                {
                    "rank": 1,
                    "node": {
                        "qualname": "molpy.typifier.atomistic.OplsAtomisticTypifier",
                        "name": "OplsAtomisticTypifier",
                        "kind": "class",
                        "signature": "OplsAtomisticTypifier()",
                        "summary": "Assign OPLS-AA atom types to an Atomistic.",
                    },
                }
            ]
        }

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="molmcp_find_capability",
                        args={"task": "assign OPLS-AA atom types"},
                    )
                ]
            )
        fix = RepairDecision(
            diagnosis="impl referenced a non-existent module; switching to OplsAtomisticTypifier",
            impl=TaskImplDraft(
                imports=("from molpy.typifier.atomistic import OplsAtomisticTypifier",),
                body="typed = OplsAtomisticTypifier()(ctx.inputs['atomistic'])",
            ),
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=fix.model_dump(mode="json"),
                )
            ]
        )

    agent = _build_repair_agent(FunctionModel(model_fn), tools=(molmcp_find_capability,))
    prompt = (
        "task_id: type\n\n"
        "--- pytest traceback ---\n"
        "AttributeError: module 'molpy' has no attribute 'forcefield'\n"
    )

    result = await agent.run(prompt)

    assert isinstance(result.output, RepairDecision)
    assert find_calls, "repair agent must invoke molmcp_find_capability on AttributeError"
    assert result.output.impl is not None
    assert any("OplsAtomisticTypifier" in line for line in result.output.impl.imports)


def test_build_repair_callable_returns_callable_or_none(tmp_path) -> None:
    """The factory returns either a closure (molmcp seeded) or None (caller
    falls back to the no-tool router path) — both shapes are valid."""
    from pydantic_ai.models.test import TestModel

    repair = build_repair_callable(workspace=tmp_path, model=TestModel())
    assert repair is None or callable(repair)


@pytest.mark.asyncio
async def test_repair_callable_respects_wall_clock_timeout(tmp_path) -> None:
    """A repair that overruns ``timeout_seconds`` raises ``TimeoutError``
    so the caller can record a failed iteration instead of hanging.

    The smoke v23 run had a 73-minute idle gap traced to the repair
    agent stuck in an MCP tool-call loop with no wall-clock bound; the
    ``asyncio.wait_for`` wrapper cancels the inner call and the caller
    treats it as a failed debug-loop iteration.
    """
    from molexp.agent._pydanticai.debug_repair import _wrap_agent_as_callable

    # Build a minimal Agent that simulates a stuck MCP loop by sleeping
    # longer than the deadline. Output type isn't exercised — the
    # asyncio.wait_for layer cancels before the agent ever returns.
    class _SlowAgent:
        def __init__(self) -> None:
            self.canceled = False

        async def __aenter__(self) -> _SlowAgent:
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        async def run(self, prompt: str, *, usage_limits: object) -> object:
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                self.canceled = True
                raise
            raise AssertionError("agent ran to completion despite timeout")

    slow = _SlowAgent()
    repair = _wrap_agent_as_callable(
        slow,  # type: ignore[arg-type]
        request_limit=10,
        timeout_seconds=0.1,
    )
    with pytest.raises(TimeoutError):
        await repair("doesn't matter")
    assert slow.canceled, "asyncio.wait_for must cancel the inner agent.run"


@pytest.mark.asyncio
async def test_repair_callable_serialises_concurrent_calls(tmp_path) -> None:
    """``_silence_process_stdio`` mutates process-global fds 1/2; concurrent
    callers would race on the dup/restore. ``_wrap_agent_as_callable``
    serialises with an internal ``asyncio.Lock`` so two ``asyncio.gather``-
    fanned calls observe each other strictly sequentially."""
    from molexp.agent._pydanticai.debug_repair import _wrap_agent_as_callable

    in_flight = 0
    max_concurrent = 0

    class _SerialAgent:
        async def __aenter__(self) -> _SerialAgent:
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        async def run(self, prompt: str, *, usage_limits: object) -> object:
            nonlocal in_flight, max_concurrent
            in_flight += 1
            max_concurrent = max(max_concurrent, in_flight)
            try:
                await asyncio.sleep(0.05)
            finally:
                in_flight -= 1

            class _Result:
                output = RepairDecision(diagnosis="ok")

            return _Result()

    agent = _SerialAgent()
    repair = _wrap_agent_as_callable(
        agent,  # type: ignore[arg-type]
        request_limit=10,
        timeout_seconds=5.0,
    )
    await asyncio.gather(*[repair(f"p{i}") for i in range(4)])
    assert max_concurrent == 1, (
        "stdio-mutating repair callable must serialise concurrent invocations"
    )


@pytest.mark.asyncio
async def test_repair_callable_aexit_grace_counted_in_budget(tmp_path) -> None:
    """Production ``MCPServer.__aexit__`` has its own shutdown grace. The
    wrapper's ``asyncio.wait_for`` cancels the inner ``agent.run`` but
    Python waits for the cancelled task's unwind (including ``__aexit__``)
    before raising ``TimeoutError``. Make sure a slow ``__aexit__`` still
    surfaces as a ``TimeoutError`` rather than hanging the caller."""
    import contextlib

    from molexp.agent._pydanticai.debug_repair import _wrap_agent_as_callable

    aexit_completed = False

    class _SlowAexitAgent:
        async def __aenter__(self) -> _SlowAexitAgent:
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            nonlocal aexit_completed
            # Shield against cancellation propagating through cleanup the
            # way pydantic-ai's MCP shutdown does via anyio.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0.05)
            aexit_completed = True

        async def run(self, prompt: str, *, usage_limits: object) -> object:
            await asyncio.sleep(5.0)
            raise AssertionError("agent.run should have been cancelled")

    agent = _SlowAexitAgent()
    repair = _wrap_agent_as_callable(
        agent,  # type: ignore[arg-type]
        request_limit=10,
        timeout_seconds=0.1,
    )
    with pytest.raises(TimeoutError):
        await repair("test")
    assert aexit_completed, (
        "wrapper must let __aexit__ finish so MCP subprocess teardown completes"
    )
