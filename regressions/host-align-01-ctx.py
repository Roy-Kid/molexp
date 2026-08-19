"""Public-API regression for host-align-01-ctx.

Hard-coded: compose_plan publishes ctx.tools; dump_config services include
``tools`` and ``artifacts``. No third-party runtime.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from molexp.harness.host import compose_plan
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
            model="none",
        )


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="host-align-01-ctx-"))
    host = compose_plan(run_id="abcd1234", run_dir=root, gateway=_Gw())
    try:
        belt = host.ctx.tools
        assert belt is not None
        services = host.dump_config()["services"]
        assert "tools" in services
        assert "artifacts" in services
    finally:
        host.unload()
    print("host-align-01-ctx: ok")


if __name__ == "__main__":
    main()
