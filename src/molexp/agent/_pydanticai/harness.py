"""``PydanticAIHarness`` — sole pydantic-ai import site for ``molexp.agent``.

Wraps a single ``pydantic_ai.Agent`` lazily and exposes a narrow
``complete`` method consumed by every ``AgentMode``. Any future
streaming or tool-loop iteration goes through this class — no other
file in ``src/molexp/agent/`` may import ``pydantic_ai``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, models
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import Tool

# pydantic-ai SDK surface types reach into the harness as opaque
# pass-through values. The aliases below pin each boundary position to
# the real SDK type rather than ``object`` so callers see the actual
# shape and ty can match overloads.
type PydanticAiModel = "models.Model | models.KnownModelName | str | None"
type PydanticAiTool = "Tool[None]"
type PydanticAiMessage = "ModelMessage"
# ``Agent.run`` returns a generic ``AgentRunResult[OutputDataT]``. The
# harness exposes only ``getattr(result, "output", ...)`` — the concrete
# type parameter is irrelevant to molexp callers, so the result is held
# as the generic ``AgentRunResult[Any]`` (the only ``Any`` site allowed
# inside the harness — pydantic-ai's generic surface forces it).
type PydanticAiRunResult = "AgentRunResult[Any]"

# Concrete agent shape: NoneType deps, str output (the harness's
# narrow ``complete`` is the only consumer and it stringifies the
# output anyway).
type _HarnessAgent = "Agent[None, str]"


class HarnessResult(BaseModel):
    """Normalized outcome of one ``harness.complete(...)`` call."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    text: str
    raw: PydanticAiRunResult | None = None


class PydanticAIHarness:
    """Sole instantiation site for ``pydantic_ai.Agent`` in ``molexp.agent``.

    Construction is cheap and side-effect-free — the underlying
    ``pydantic_ai.Agent`` is built on first :meth:`complete` call.
    """

    def __init__(
        self,
        *,
        model: PydanticAiModel,
        tools: tuple[PydanticAiTool, ...] = (),
        system_prompt: str = "",
        workspace: Path | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.workspace = workspace
        self._agent: _HarnessAgent | None = None

    def _ensure_agent(self) -> _HarnessAgent:
        if self._agent is None:
            if self.system_prompt and self.tools:
                self._agent = Agent(
                    model=self.model,
                    system_prompt=self.system_prompt,
                    tools=list(self.tools),
                )
            elif self.system_prompt:
                self._agent = Agent(
                    model=self.model,
                    system_prompt=self.system_prompt,
                )
            elif self.tools:
                self._agent = Agent(model=self.model, tools=list(self.tools))
            else:
                self._agent = Agent(model=self.model)
        return self._agent

    async def complete(
        self,
        prompt: str,
        *,
        message_history: tuple[PydanticAiMessage, ...] = (),
    ) -> HarnessResult:
        """Drive one ``pydantic_ai.Agent.run`` round-trip and normalize the result."""

        agent = self._ensure_agent()
        run_result = await agent.run(
            prompt,
            message_history=list(message_history) if message_history else None,
        )
        text = str(getattr(run_result, "output", "") or "")
        return HarnessResult(text=text, raw=run_result)


__all__ = ["HarnessResult", "PydanticAIHarness"]
