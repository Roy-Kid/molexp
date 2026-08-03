"""``import molexp.server.app`` must have no side effects.

The module used to end with ``app = create_app(serve_static=False)`` for
``uvicorn --reload``. Building an app at import time meant *importing* the
module read ``~/.molexp/config.json`` from disk and bridged it into the
process-global ``molexp.config`` — so a library user who merely imported the
server picked up the operator's model and API key, and any test that imported
it (directly or through a conftest) silently re-tiered every later test in the
same process. The dev entry point is the factory form
(``uvicorn --factory molexp.server.app:create_app``).
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_app_module_does_not_touch_global_config() -> None:
    code = (
        "import molexp\n"
        "import molexp.server.app\n"
        "assert dict(molexp.config) == {}, "
        "f'importing molexp.server.app mutated molexp.config: {dict(molexp.config)}'\n"
        "print('clean')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_app_module_exposes_no_prebuilt_app_attribute() -> None:
    import molexp.server.app as app_module

    assert not hasattr(app_module, "app"), (
        "a module-level `app` re-introduces import-time app construction; "
        "serve via `uvicorn --factory molexp.server.app:create_app`"
    )
