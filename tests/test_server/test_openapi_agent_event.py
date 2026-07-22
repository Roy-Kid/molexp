"""The ``scripts/dump_openapi.py`` regen is byte-stable and server-boot-free (spec 01).

The dump feeds ``npm run generate:api`` (link 02), so its output must be
deterministic and the AgentEvent vocabulary must be present in the emitted
schema text.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dump_openapi.py"


def _load_dump():
    spec = importlib.util.spec_from_file_location("dump_openapi", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.dump_openapi


def test_dump_openapi_is_deterministic_and_bootless(tmp_path: Path) -> None:
    dump_openapi = _load_dump()
    a = dump_openapi(tmp_path / "a.json")
    b = dump_openapi(tmp_path / "b.json")
    assert a.read_bytes() == b.read_bytes(), "two dumps must be byte-identical"
    # sorted-key JSON (determinism marker) and non-trivial content
    text = a.read_text(encoding="utf-8")
    assert '"openapi"' in text
    assert "ThinkingDeltaEvent" in text
