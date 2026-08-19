"""Public-API regression for host-align-04-plugins."""

from __future__ import annotations

import tempfile
from pathlib import Path

from molexp.harness.host import compose_run


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="host-align-04-plugins-"))
    host = compose_run(run_id="abcd1234", run_dir=root)
    try:
        assert host.ctx.workspace is not None
        assert host.ctx.workflow is not None
        assert "workspace" in host.dump()
        assert "workflow" in host.dump()
    finally:
        host.unload()
    assert hasattr(host.ctx, "workspace") is False
    print("host-align-04-plugins: ok")


if __name__ == "__main__":
    main()
