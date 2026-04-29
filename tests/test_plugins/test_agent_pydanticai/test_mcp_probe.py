"""Unit tests for the MCP probe error formatter.

Focus: ``_format_error`` must surface the *root* cause of a failed probe,
not the outermost ``ExceptionGroup``-style wrapper anyio adds. Without
unwrapping, users would see "unhandled errors in a TaskGroup" for every
401, 5xx, or DNS issue.
"""

from __future__ import annotations

import pytest

from molexp.plugins.agent_pydanticai.mcp_probe import _format_error, _innermost


def test_format_error_unwraps_exception_group_to_inner_cause():
    inner = ValueError("HTTP 401 Unauthorized")
    outer: BaseException = ExceptionGroup("oh no", [inner])
    msg = _format_error(outer)
    assert msg == "ValueError: HTTP 401 Unauthorized"


def test_format_error_walks_explicit_cause_chain():
    try:
        try:
            raise ConnectionError("dns lookup failed")
        except ConnectionError as e:
            raise RuntimeError("client init") from e
    except RuntimeError as exc:
        msg = _format_error(exc)
    assert "dns lookup failed" in msg


def test_format_error_handles_plain_exception():
    msg = _format_error(KeyError("missing"))
    assert msg.startswith("KeyError")
    assert "missing" in msg


def test_format_error_truncates_overly_long_messages():
    long = "x" * 1000
    msg = _format_error(ValueError(long))
    assert len(msg) <= 500  # leeway for class-name prefix
    assert msg.endswith("…")


def test_format_error_falls_back_to_class_name_when_message_empty():
    class Boom(Exception):
        pass

    assert _format_error(Boom()) == "Boom: Boom"


def test_innermost_stops_on_self_referential_chain():
    """Defensive — pathological code can produce ``__context__`` cycles."""
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__context__ = b
    b.__context__ = a
    # Should terminate (depth-bounded), not recurse forever.
    leaf = _innermost(a)
    assert isinstance(leaf, RuntimeError)


def test_format_error_picks_first_subexception_in_group():
    eg: BaseException = ExceptionGroup(
        "outer",
        [TimeoutError("read timeout"), ConnectionError("reset by peer")],
    )
    msg = _format_error(eg)
    # Order is "first in list" — picks TimeoutError, not the second one.
    assert "TimeoutError" in msg
    assert "read timeout" in msg
