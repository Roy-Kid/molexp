"""AgentEvent union is first-class in OpenAPI + a deterministic dump (spec 01).

The streamed event vocabulary must appear `kind`-discriminated in the schema so
``npm run generate:api`` (link 02) can emit narrowed TS models, and the
``scripts/dump_openapi.py`` regen must be byte-stable and server-boot-free.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from molexp.server.app import create_app

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dump_openapi.py"


def _load_dump():
    spec = importlib.util.spec_from_file_location("dump_openapi", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.dump_openapi


def test_schema_contains_kind_discriminated_agent_event_union() -> None:
    schemas = create_app().openapi()["components"]["schemas"]
    # the reasoning + answer + tool member models are present, keyed on `kind`
    for member in (
        "ThinkingDeltaEvent",
        "TokenDeltaEvent",
        "ToolCallStartedEvent",
        "ToolCallCompletedEvent",
        "ModeCompletedEvent",
    ):
        assert member in schemas, f"{member} missing from OpenAPI components"
        kind = schemas[member]["properties"]["kind"]
        # a pinned Literal renders as const (or a single-value enum)
        assert kind.get("const") or kind.get("enum"), f"{member}.kind is not a discriminator"


def test_dump_openapi_is_deterministic_and_bootless(tmp_path: Path) -> None:
    dump_openapi = _load_dump()
    a = dump_openapi(tmp_path / "a.json")
    b = dump_openapi(tmp_path / "b.json")
    assert a.read_bytes() == b.read_bytes(), "two dumps must be byte-identical"
    # sorted-key JSON (determinism marker) and non-trivial content
    text = a.read_text(encoding="utf-8")
    assert '"openapi"' in text
    assert "ThinkingDeltaEvent" in text
