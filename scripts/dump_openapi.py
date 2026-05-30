#!/usr/bin/env python
"""Dump the molexp API OpenAPI schema to repo-root ``openapi.json``.

The UI's ``npm run generate:api`` consumes ``openapi.json`` (an untracked,
generated artifact). This script regenerates it deterministically so the
schema → TypeScript codegen pipeline is reproducible and CI-checkable.

Deterministic + boot-free: it constructs the FastAPI app in-process and calls
``app.openapi()`` (no uvicorn, no network), writing sorted-key JSON so two runs
are byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "openapi.json"


def dump_openapi(path: Path | None = None) -> Path:
    """Write the OpenAPI schema to ``path`` (default repo-root ``openapi.json``).

    Args:
        path: Destination file; defaults to the repo-root ``openapi.json`` the
            UI codegen reads.

    Returns:
        The path written. The content is ``sort_keys``-stable JSON with a
        trailing newline, generated without booting a server or touching the
        network.
    """
    from molexp.server.app import create_app

    schema = create_app().openapi()
    out = path or _DEFAULT_OUT
    out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    written = dump_openapi()
    print(f"wrote {written}")
