"""Source-grounded debug-loop repair agent — offline tests.

`_build_repair_agent` wires a `pydantic_ai.Agent[None, GeneratedModule]`
with `toolsets=[MCPToolset(...)]` (or injectable `tools=` for tests).
`build_repair_callable` returns a closure that drives it once per call,
or `None` when no molmcp `stdio` entry is configured.

These tests use `FunctionModel` + fake source-introspection callables —
no live LLM, no real MCP subprocess.
"""

from __future__ import annotations

import pytest

from molexp.agent._pydanticai.debug_repair import (
    _build_repair_agent,
    build_repair_callable,
)
from molexp.agent.modes.author.codegen import GeneratedModule

pytest.importorskip("pydantic_ai")


def test_build_repair_agent_returns_pydantic_ai_agent() -> None:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    assert isinstance(_build_repair_agent(TestModel()), Agent)


@pytest.mark.asyncio
async def test_repair_agent_invokes_get_source_on_attribute_error() -> None:
    """The MCP-attached repair agent uses `get_source` to find the real API
    before patching, given an `AttributeError`-style traceback."""
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    get_source_calls: list[str] = []

    async def get_source(path: str) -> str:
        """Fake molmcp source-read — shows a `__init__.py` re-export."""
        get_source_calls.append(path)
        return "from .core.atomistic import Atomistic\n"

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
                parts=[ToolCallPart(tool_name="get_source", args={"path": "molpy/__init__.py"})]
            )
        fix = GeneratedModule(
            task_id="build",
            source="from molpy.core.atomistic import Atomistic\nVALUE = 2\n",
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=fix.model_dump(mode="json"),
                )
            ]
        )

    agent = _build_repair_agent(FunctionModel(model_fn), tools=(get_source,))
    prompt = (
        "task_id: build\n\n"
        "--- pytest traceback ---\n"
        "AttributeError: module 'molpy' has no attribute 'forcefield'\n"
    )

    result = await agent.run(prompt)

    assert isinstance(result.output, GeneratedModule)
    assert get_source_calls, "repair agent must invoke get_source on AttributeError"
    assert "from molpy.core.atomistic" in result.output.source


def test_build_repair_callable_returns_callable_or_none(tmp_path) -> None:
    """The factory returns either a closure (molmcp seeded) or None (caller
    falls back to the legacy router path) — both shapes are valid."""
    from pydantic_ai.models.test import TestModel

    repair = build_repair_callable(workspace=tmp_path, model=TestModel())
    assert repair is None or callable(repair)
