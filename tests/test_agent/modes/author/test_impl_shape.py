"""Tests for the impl-codegen structural validator (``_check_impl_shape``).

The validator enforces a tight surface: an impl module may contain only

- an optional module docstring,
- imports (``import X`` / ``from X import Y``),
- exactly one top-level ``async def <task>(ctx)`` function.

Anything else — a class, a top-level assignment, an extra def, a test
definition — is rejected. This eliminates the surface for module-level
monkey-patching, stray helper functions, and pytest test code in the
impl path.
"""

from __future__ import annotations

import pytest

from molexp.agent.modes.author.codegen import _check_impl_shape


def test_valid_async_function_passes() -> None:
    source = (
        '"""docstring"""\n'
        "from molpy import System\n"
        "\n"
        "\n"
        "async def step_1_build_chain(ctx):\n"
        "    chain = System()\n"
        "    return {'chain': chain}\n"
    )
    assert _check_impl_shape(source) is None


def test_module_with_no_task_function_is_rejected() -> None:
    source = "from molpy import System\n"
    issue = _check_impl_shape(source)
    assert issue is not None
    assert "no `async def <task>(ctx)`" in issue


def test_module_with_sync_def_is_rejected() -> None:
    """A sync ``def`` at module level is banned (must be async)."""
    source = "def helper(x):\n    return x + 1\n"
    issue = _check_impl_shape(source)
    assert issue is not None
    assert "def helper" in issue


def test_module_with_class_definition_is_rejected() -> None:
    """No classes at module level — the function shape has no room for them."""
    source = "class Helper:\n    pass\n\n\nasync def task(ctx):\n    return None\n"
    issue = _check_impl_shape(source)
    assert issue is not None
    assert "class Helper" in issue


def test_module_level_attribute_assignment_is_rejected() -> None:
    """Module-level monkey-patching: ``SomeClass.attr = ...`` — banned."""
    source = (
        "from molpy.core.atomistic import Atom\n"
        "\n"
        "Atom.symbol = property(lambda self: self['element'])\n"
        "\n"
        "\n"
        "async def task(ctx):\n"
        "    return None\n"
    )
    issue = _check_impl_shape(source)
    assert issue is not None
    assert "Atom.symbol" in issue


def test_module_level_plain_assignment_is_rejected() -> None:
    """Even a bare ``X = 1`` at module level isn't allowed — the shape
    has imports + the one function, nothing else."""
    source = "X = 1\n\n\nasync def task(ctx):\n    return None\n"
    issue = _check_impl_shape(source)
    assert issue is not None
    assert "X = ..." in issue


def test_multiple_async_functions_rejected() -> None:
    source = (
        "async def task_a(ctx):\n    return None\n\n\nasync def task_b(ctx):\n    return None\n"
    )
    issue = _check_impl_shape(source)
    assert issue is not None
    assert "multiple" in issue


def test_async_test_function_rejected() -> None:
    """A test-shaped async function in the impl path is rejected."""
    source = "async def test_runs(ctx):\n    return None\n"
    issue = _check_impl_shape(source)
    assert issue is not None
    assert "looks like a test" in issue


def test_wrong_signature_rejected() -> None:
    """``async def task(self)`` is rejected — the parameter must be ``ctx``."""
    source = "async def task(self):\n    return None\n"
    issue = _check_impl_shape(source)
    assert issue is not None
    assert "wrong signature" in issue


def test_unparseable_source_raises() -> None:
    """Caller is expected to have run ``validate_python`` first; passing
    unparseable source to the shape check raises ``SyntaxError``."""
    with pytest.raises(SyntaxError):
        _check_impl_shape("def x(:\n")
