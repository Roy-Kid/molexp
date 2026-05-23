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
async def test_repair_agent_invokes_find_capability_on_attribute_error() -> None:
    """The MCP-attached repair agent invokes `molmcp_find_capability` to
    locate the real API for the missing symbol before patching the impl —
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
        fix = GeneratedModule(
            task_id="type",
            source=(
                "from molpy.typifier.atomistic import OplsAtomisticTypifier\n"
                "# ... use the real API ...\n"
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

    assert isinstance(result.output, GeneratedModule)
    assert find_calls, "repair agent must invoke molmcp_find_capability on AttributeError"
    assert "OplsAtomisticTypifier" in result.output.source


def test_build_repair_callable_returns_callable_or_none(tmp_path) -> None:
    """The factory returns either a closure (molmcp seeded) or None (caller
    falls back to the legacy router path) — both shapes are valid."""
    from pydantic_ai.models.test import TestModel

    repair = build_repair_callable(workspace=tmp_path, model=TestModel())
    assert repair is None or callable(repair)
