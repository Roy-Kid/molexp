"""Public-API regression for host-reflect-01-plugin."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from molexp.harness.host import Host, Keys
from molexp.harness.host.plugins.agent_call import AgentCallPlugin, AgentStep
from molexp.harness.host.plugins.reflection import Reflection
from molexp.harness.schemas import AgentCallResult, AgentCallSpec, PlanArtifactRef


class _Gw:
    async def call(self, spec: AgentCallSpec) -> AgentCallResult:
        del spec
        dummy = PlanArtifactRef(
            id="art-1",
            kind="log",
            uri="memory://art-1",
            sha256="a" * 64,
            created_at=datetime.now(tz=UTC),
            created_by="reg",
        )
        return AgentCallResult(
            output_artifact=dummy,
            raw_response_artifact=dummy,
            model="base",
        )


async def _main() -> None:
    host = Host()
    host.ctx.provide(Keys.ARTIFACTS, object())
    host.mount(AgentCallPlugin(_Gw()))

    async def critic(step: AgentStep) -> AgentCallResult:
        return step.result.model_copy(update={"model": "reflected"})

    host.mount(Reflection(critic=critic))
    result = await host.ctx.llm.call(  # type: ignore[union-attr]
        AgentCallSpec(agent_name="orig", input_artifact_ids=[], output_schema={})
    )
    assert result.model == "reflected"
    print("host-reflect-01-plugin: ok")


if __name__ == "__main__":
    asyncio.run(_main())
