"""Built-in workflow templates the agent can attach to experiments.

These exist so the agent can drive the full workspace lifecycle from chat
without the user having to write Python files. Each template is a tiny
``RunContext``-style callable; the ``set_workflow`` call promotes it to a
single-task ``WorkflowSpec`` automatically.

Add new templates here by:
1. Writing the callable (``def fn(ctx: me.RunContext) -> None``)
2. Appending an entry to :data:`TEMPLATES`

Keep them deterministic, fast, and free of domain-specific dependencies —
they exist for end-to-end smoke testing of the agent ↔ workspace ↔ workflow
loop, not for science.
"""

from __future__ import annotations

from collections.abc import Callable

import molexp as me


def _square(ctx: me.RunContext) -> None:
    """Compute ``y = x ** 2`` from a single numeric parameter ``x``."""
    x = float(ctx.params.get("x", 0))
    y = x * x
    ctx.set_result("x", x)
    ctx.set_result("y", y)
    ctx.log("compute").append(f"square({x}) = {y}")


def _cube(ctx: me.RunContext) -> None:
    """Compute ``y = x ** 3`` from a single numeric parameter ``x``."""
    x = float(ctx.params.get("x", 0))
    y = x ** 3
    ctx.set_result("x", x)
    ctx.set_result("y", y)
    ctx.log("compute").append(f"cube({x}) = {y}")


def _add(ctx: me.RunContext) -> None:
    """Compute ``z = a + b`` from two numeric parameters ``a`` and ``b``."""
    a = float(ctx.params.get("a", 0))
    b = float(ctx.params.get("b", 0))
    z = a + b
    ctx.set_result("a", a)
    ctx.set_result("b", b)
    ctx.set_result("z", z)
    ctx.log("compute").append(f"{a} + {b} = {z}")


# Public registry: ``{name: (callable, human_description, expected_params)}``.
TEMPLATES: dict[str, tuple[Callable[..., None], str, list[str]]] = {
    "square": (_square, "Compute y = x^2 from numeric parameter `x`.", ["x"]),
    "cube": (_cube, "Compute y = x^3 from numeric parameter `x`.", ["x"]),
    "add": (_add, "Compute z = a + b from numeric parameters `a` and `b`.", ["a", "b"]),
}


def get_template(name: str) -> tuple[Callable[..., None], str, list[str]] | None:
    """Look up a template by name; returns ``None`` when unknown."""
    return TEMPLATES.get(name)


def list_templates() -> list[dict[str, object]]:
    """Render every template as a JSON-serializable dict for the agent."""
    return [
        {"name": name, "description": desc, "parameters": list(params)}
        for name, (_, desc, params) in TEMPLATES.items()
    ]
