"""Tests for :mod:`molexp.agent._pydanticai.retry`.

Acceptance criterion ac-003: ``RetryPolicy`` rejects bad inputs at
construction; ``sleep_for`` matches ``backoff * 2 ** (attempt-1)``;
``should_retry`` returns ``False`` once attempt reaches the cap.
"""

from __future__ import annotations

import pydantic
import pytest
from pydantic import ValidationError

import molexp.agent._pydanticai.retry as retry_mod
from molexp.agent._pydanticai.errors import ErrorKind, classify
from molexp.agent._pydanticai.retry import RetryPolicy, should_retry, sleep_for

# ── RetryPolicy validation ─────────────────────────────────────────────────


def test_retry_policy_default_construction() -> None:
    policy = RetryPolicy()
    assert policy.max_attempts == 3
    assert policy.backoff_seconds == 0.5
    assert ErrorKind.model_unavailable in policy.retry_on
    assert ErrorKind.timeout in policy.retry_on
    # schema_parse is deliberately NOT in the default retry_on after
    # ``plan-mode-pydanticai-rewrite``: pydantic-ai's Agent(output_retries=N)
    # retries schema-parse failures at the model level with the
    # validation error fed back as a cheap follow-up turn; the router
    # used to re-issue the full prompt on the same kind of failure and
    # burned 14:30 min on one call in production.
    assert ErrorKind.schema_parse not in policy.retry_on
    assert ErrorKind.validation not in policy.retry_on
    assert ErrorKind.unknown not in policy.retry_on


def test_retry_policy_rejects_max_attempts_below_one() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=-3)


def test_retry_policy_rejects_negative_backoff() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(backoff_seconds=-0.1)


def test_retry_policy_zero_backoff_is_allowed() -> None:
    policy = RetryPolicy(backoff_seconds=0.0)
    assert policy.backoff_seconds == 0.0


def test_retry_policy_is_frozen() -> None:
    policy = RetryPolicy()
    with pytest.raises(ValidationError):
        policy.max_attempts = 5  # type: ignore[misc]


def test_retry_policy_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(unknown=1)  # type: ignore[call-arg]


# ── should_retry ───────────────────────────────────────────────────────────


def test_should_retry_returns_false_when_attempt_at_cap() -> None:
    policy = RetryPolicy(max_attempts=3)
    # The 3rd attempt (attempt=3) is the last one; should not retry.
    assert should_retry(ErrorKind.timeout, policy, attempt=3) is False


def test_should_retry_returns_false_when_attempt_above_cap() -> None:
    policy = RetryPolicy(max_attempts=2)
    assert should_retry(ErrorKind.timeout, policy, attempt=5) is False


def test_should_retry_true_when_kind_in_retry_on_and_attempts_left() -> None:
    policy = RetryPolicy(max_attempts=3)
    assert should_retry(ErrorKind.timeout, policy, attempt=1) is True
    assert should_retry(ErrorKind.timeout, policy, attempt=2) is True


def test_should_retry_false_when_kind_not_in_retry_on() -> None:
    policy = RetryPolicy(max_attempts=3)
    assert should_retry(ErrorKind.unknown, policy, attempt=1) is False
    assert should_retry(ErrorKind.validation, policy, attempt=1) is False


def test_should_retry_with_max_attempts_one_never_retries() -> None:
    policy = RetryPolicy(max_attempts=1)
    assert should_retry(ErrorKind.timeout, policy, attempt=1) is False


# ── sleep_for ──────────────────────────────────────────────────────────────


def test_sleep_for_exponential_backoff() -> None:
    policy = RetryPolicy(backoff_seconds=0.5)
    assert sleep_for(policy, attempt=1) == pytest.approx(0.5)
    assert sleep_for(policy, attempt=2) == pytest.approx(1.0)
    assert sleep_for(policy, attempt=3) == pytest.approx(2.0)
    assert sleep_for(policy, attempt=4) == pytest.approx(4.0)


def test_sleep_for_zero_backoff_yields_zero() -> None:
    policy = RetryPolicy(backoff_seconds=0.0)
    assert sleep_for(policy, attempt=1) == 0.0
    assert sleep_for(policy, attempt=5) == 0.0


def test_sleep_for_custom_base() -> None:
    policy = RetryPolicy(backoff_seconds=0.25)
    assert sleep_for(policy, attempt=3) == pytest.approx(1.0)


# ── Docstring ownership-split enumeration (ac-003) ─────────────────────────


def _combined_retry_docstrings() -> str:
    """Module + ``RetryPolicy`` docstrings, lower-cased for substring checks."""
    module_doc = retry_mod.__doc__ or ""
    policy_doc = RetryPolicy.__doc__ or ""
    return (module_doc + "\n" + policy_doc).lower()


def test_retry_docstrings_enumerate_output_retries_owner() -> None:
    """ac-003 — the docstrings must name pydantic-ai output_retries as
    the owner of output/parse-validation retries.

    RED today: current docstrings mention ``output_retries`` only in
    the ``retry_on`` note but the full ownership enumeration / the
    other owners below are absent.
    """
    doc = _combined_retry_docstrings()
    assert "output_retries" in doc


def test_retry_docstrings_enumerate_agent_retries_owner() -> None:
    """ac-003 — the docstrings must name pydantic-ai ``Agent(retries=)``
    as the owner of ModelRetry/tool-validation retries.

    RED today: current docstrings do not mention ``Agent(retries=``.
    """
    doc = _combined_retry_docstrings()
    assert "agent(retries=" in doc


def test_retry_docstrings_enumerate_transport_shim_owner() -> None:
    """ac-003 — the docstrings must state the router transport shim owns
    ONLY model_unavailable/timeout."""
    doc = _combined_retry_docstrings()
    assert "transport" in doc
    assert "model_unavailable" in doc
    assert "timeout" in doc


def test_retry_docstrings_record_async_tenacity_transport_non_adoption() -> None:
    """ac-003 — the docstrings must record that
    ``AsyncTenacityTransport`` exists but is deliberately NOT adopted,
    with a rationale referencing caller-supplied / opaque ``Model``
    instances.

    RED today: current docstrings never mention ``AsyncTenacityTransport``.
    """
    doc = _combined_retry_docstrings()
    assert "asynctenacitytransport" in doc
    # Non-adoption rationale: must reference the opaque / caller-supplied
    # Model injection contract.
    assert "model" in doc
    assert ("opaque" in doc) or ("caller-supplied" in doc) or ("caller supplied" in doc)


# ── errors.py taxonomy lock (ac-007) ───────────────────────────────────────


def test_error_kind_member_set_unchanged() -> None:
    """ac-007 — the ErrorKind member set is exactly the documented five."""
    assert {kind.name for kind in ErrorKind} == {
        "model_unavailable",
        "schema_parse",
        "validation",
        "timeout",
        "unknown",
    }


def test_classify_maps_pydantic_validation_error_to_schema_parse() -> None:
    """ac-007 — pydantic.ValidationError -> schema_parse."""

    class _M(pydantic.BaseModel):
        x: int

    try:
        _M(x="not an int")  # type: ignore[arg-type]
    except pydantic.ValidationError as exc:
        assert classify(exc) is ErrorKind.schema_parse
    else:  # pragma: no cover - construction must fail
        pytest.fail("expected a pydantic.ValidationError")


def test_classify_maps_timeout_error_to_timeout() -> None:
    """ac-007 — TimeoutError -> timeout (``asyncio.TimeoutError`` is the same
    object as the builtin ``TimeoutError`` since 3.11, so one assertion covers
    both)."""
    assert classify(TimeoutError()) is ErrorKind.timeout


def test_classify_maps_oserror_to_model_unavailable() -> None:
    """ac-007 — OSError (and its ConnectionError subclass) ->
    model_unavailable."""
    assert classify(OSError()) is ErrorKind.model_unavailable
    assert classify(ConnectionError("refused")) is ErrorKind.model_unavailable


def test_classify_maps_type_error_to_schema_parse() -> None:
    """ac-007 — TypeError (the router's own schema-mismatch raise) ->
    schema_parse."""
    assert classify(TypeError("wrong type")) is ErrorKind.schema_parse


def test_classify_maps_unrecognized_exception_to_unknown() -> None:
    """ac-007 — an unrecognized exception -> unknown.

    Plain ``ValueError`` / ``RuntimeError`` are NOT
    ``pydantic.ValidationError`` and match no other branch, so they
    fall through to ``unknown``."""
    assert classify(ValueError("plain")) is ErrorKind.unknown
    assert classify(RuntimeError("plain")) is ErrorKind.unknown
