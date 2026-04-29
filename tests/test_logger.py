"""Tests for molexp's Logger subclass and get_logger factory."""

from __future__ import annotations

from typing import Any

import mollog
import pytest

import molexp
from molexp._logger import _reset_cache


@pytest.fixture(autouse=True)
def _isolate_logger_cache() -> Any:
    """Each test gets a fresh molexp logger cache."""
    _reset_cache()
    yield
    _reset_cache()


class _CapturingHandler(mollog.Handler):
    """Test handler that records every dispatched LogRecord."""

    def __init__(self) -> None:
        super().__init__(level=mollog.Level.TRACE)
        self.records: list[mollog.LogRecord] = []

    def emit(self, record: mollog.LogRecord) -> None:
        self.records.append(record)


def test_get_logger_returns_molexp_logger_subclass() -> None:
    log = molexp.get_logger("molexp.test.subclass")
    assert isinstance(log, molexp.Logger)
    assert isinstance(log, mollog.Logger)


def test_get_logger_is_cached_per_name() -> None:
    a = molexp.get_logger("molexp.test.cache")
    b = molexp.get_logger("molexp.test.cache")
    assert a is b


def test_inherited_verbs_still_work() -> None:
    handler = _CapturingHandler()
    log = molexp.get_logger("molexp.test.inherited")
    log.add_handler(handler)

    log.info("classic", foo=1)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.level is mollog.Level.INFO
    assert record.message == "classic"
    assert record.extra.get("foo") == 1
    assert "verb" not in record.extra


def test_ice_emits_tagged_record() -> None:
    handler = _CapturingHandler()
    log = molexp.get_logger("molexp.test.ice")
    log.add_handler(handler)

    log.ice("agent step", agent_id="a-1", step=3)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.level is mollog.Level.INFO
    assert record.message == "agent step"
    assert record.extra.get("verb") == "ice"
    assert record.extra.get("agent_id") == "a-1"
    assert record.extra.get("step") == 3


def test_ice_without_fields_still_tags_verb() -> None:
    handler = _CapturingHandler()
    log = molexp.get_logger("molexp.test.ice.bare")
    log.add_handler(handler)

    log.ice("bare")

    assert handler.records[0].extra == {"verb": "ice"}


def test_import_molexp_does_not_mutate_mollog_logger() -> None:
    """Plugin must not monkey-patch the upstream Logger class."""
    assert not hasattr(mollog.Logger, "ice")


def test_parent_chain_propagates_to_mollog_root() -> None:
    """Records dispatch up to mollog's root handler too."""
    root_handler = _CapturingHandler()
    mollog.LoggerManager().root.add_handler(root_handler)
    try:
        log = molexp.get_logger("molexp.test.propagate")
        log.ice("hello")
        assert any(r.message == "hello" for r in root_handler.records)
    finally:
        mollog.LoggerManager().root.remove_handler(root_handler)
