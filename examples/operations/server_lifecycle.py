"""Server lifecycle — creating the FastAPI app and inspecting `ServerManager`.

Matches ``docs/guide/server-lifecycle.md``.

Running a real server inside a documentation example is fragile: it needs a
free port, a live event loop, and somewhere to park the subprocess. So this
script instead:

1. Builds the FastAPI application directly via ``create_app()`` — the same
   entry point that ``molexp serve`` uses — and lists the registered routes.
2. Instantiates ``ServerManager`` with a temp config directory so you can
   see the PID / log file locations it would write to.
3. Prints the equivalent one-line ``molexp serve`` invocation.

Run directly::

    python examples/operations/server_lifecycle.py

For the real thing, run ``molexp serve /path/to/workspace`` in a terminal.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from molexp.server import ServerManager
from molexp.server.app import create_app


def main() -> None:
    # 1. Build the ASGI app ---------------------------------------------------
    app = create_app(serve_static=False)
    print(f"FastAPI app: {app.title} v{app.version}")
    api_routes = [r.path for r in app.routes if r.path.startswith("/api")]
    print(f"registered API routes ({len(api_routes)}):")
    for route in sorted(api_routes)[:10]:
        print(f"  {route}")
    print("  …")

    # 2. ServerManager surface ------------------------------------------------
    config_dir = Path(tempfile.mkdtemp(prefix="molexp-server-state-"))
    manager = ServerManager(config_dir=config_dir)
    print(f"\nServerManager config dir: {manager.config_dir}")
    print(f"  api pid file: {manager.pid_file}")
    print(f"  api log file: {manager.server_log}")
    print(f"  running now:  {manager.is_running()}")
    print(f"  status():     {manager.status()}")

    # 3. Equivalent CLI -------------------------------------------------------
    print("\nequivalent CLI:  molexp serve ./lab --port 8000")


if __name__ == "__main__":
    main()
