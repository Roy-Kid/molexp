"""Tests for :mod:`molexp.agent._pydanticai.prompt`.

Acceptance criterion ac-005: happy-path renders correctly; missing
keys raise :class:`ProviderError` with kind
:attr:`ErrorKind.validation`.
"""

from __future__ import annotations

import pytest

from molexp.agent._pydanticai.errors import ErrorKind, ProviderError
from molexp.agent._pydanticai.prompt import render_prompt
from molexp.agent.modes.plan.protocols import ModelTier


def test_render_prompt_happy_path() -> None:
    assert render_prompt("hello ${who}", {"who": "world"}) == "hello world"


def test_render_prompt_multiple_substitutions() -> None:
    out = render_prompt(
        "${greeting}, ${who}! ${count} times.",
        {"greeting": "Hi", "who": "Alice", "count": 3},
    )
    assert out == "Hi, Alice! 3 times."


def test_render_prompt_no_placeholders_passes_through() -> None:
    assert render_prompt("static text", {}) == "static text"


def test_render_prompt_extra_context_keys_are_ignored() -> None:
    out = render_prompt("hello ${who}", {"who": "world", "unused": "x"})
    assert out == "hello world"


def test_render_prompt_missing_key_raises_provider_error_validation() -> None:
    with pytest.raises(ProviderError) as exc_info:
        render_prompt("hello ${missing}", {})
    err = exc_info.value
    assert err.kind is ErrorKind.validation
    assert err.tier is ModelTier.DEFAULT  # default sentinel
    assert err.cause is not None
    assert isinstance(err.cause, KeyError)


def test_render_prompt_missing_key_propagates_node_id_and_tier() -> None:
    with pytest.raises(ProviderError) as exc_info:
        render_prompt(
            "hello ${missing}",
            {},
            node_id="codegen",
            tier=ModelTier.HEAVY,
        )
    err = exc_info.value
    assert err.node_id == "codegen"
    assert err.tier is ModelTier.HEAVY


def test_render_prompt_malformed_template_raises_provider_error() -> None:
    """A bare ``$`` not followed by an identifier breaks
    :meth:`Template.substitute` with a :class:`ValueError`."""
    with pytest.raises(ProviderError) as exc_info:
        render_prompt("hello $", {})
    assert exc_info.value.kind is ErrorKind.validation
