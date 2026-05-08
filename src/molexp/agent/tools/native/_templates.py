"""Built-in workflow templates exposed by ``native:create_experiment``.

These are tiny, deterministic toy workflows the agent can attach to an
experiment without the user writing Python files. They exist for
end-to-end smoke testing of the agent ↔ workspace ↔ workflow loop —
not for actual science.

Add a new template by:

1. Writing a callable with signature ``def fn(ctx: me.RunContext) -> None``.
2. Appending an entry to :data:`TEMPLATES`.
"""

from __future__ import annotations

from collections.abc import Callable

import molexp as me
from molexp._typing import JSONValue


def _param_float(ctx: me.RunContext, key: str, default: float = 0.0) -> float:
    """Read a numeric parameter as ``float``, falling back to *default*.

    Mirrors the JSON-shaped runtime: ``ctx.params`` is ``dict[str, JSONValue]``
    and we know each template's required parameter is numeric. Returns
    *default* for missing / non-numeric cells so the helper is total.
    """
    value: JSONValue = ctx.params.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _square(ctx: me.RunContext) -> None:
    """Compute ``y = x ** 2`` from a single numeric parameter ``x``."""

    x = _param_float(ctx, "x")
    y = x * x
    ctx.set_result("x", x)
    ctx.set_result("y", y)
    ctx.log("compute").append(f"square({x}) = {y}")


def _cube(ctx: me.RunContext) -> None:
    """Compute ``y = x ** 3`` from a single numeric parameter ``x``."""

    x = _param_float(ctx, "x")
    y = x**3
    ctx.set_result("x", x)
    ctx.set_result("y", y)
    ctx.log("compute").append(f"cube({x}) = {y}")


def _add(ctx: me.RunContext) -> None:
    """Compute ``z = a + b`` from two numeric parameters."""

    a = _param_float(ctx, "a")
    b = _param_float(ctx, "b")
    z = a + b
    ctx.set_result("a", a)
    ctx.set_result("b", b)
    ctx.set_result("z", z)
    ctx.log("compute").append(f"{a} + {b} = {z}")


TEMPLATES: dict[str, tuple[Callable[..., None], str, list[str]]] = {
    "square": (_square, "Compute y = x^2 from numeric parameter `x`.", ["x"]),
    "cube": (_cube, "Compute y = x^3 from numeric parameter `x`.", ["x"]),
    "add": (_add, "Compute z = a + b from numeric parameters `a` and `b`.", ["a", "b"]),
}


def list_templates() -> list[dict[str, object]]:
    """Return every template as a JSON-serializable dict for the agent."""

    return [
        {"name": name, "description": desc, "parameters": list(params)}
        for name, (_, desc, params) in TEMPLATES.items()
    ]
