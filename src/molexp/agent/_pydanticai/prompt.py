"""Lightweight prompt template renderer.

Wraps :class:`string.Template` with ``${name}`` placeholders and
normalizes :class:`KeyError` / :class:`ValueError` (raised by
:meth:`string.Template.substitute` for unknown keys / malformed
templates) into a :class:`~molexp.agent._pydanticai.errors.ProviderError`
of kind :attr:`ErrorKind.validation`.

Uses no external dependencies — Jinja2 / format strings are
out of scope. The wrapper is the public surface for prompt rendering;
agent modes never call ``string.Template`` directly so the error
channel stays uniform.
"""

from __future__ import annotations

from collections.abc import Mapping
from string import Template
from typing import Any

from molexp.agent.modes.plan.protocols import ModelTier

from .errors import ErrorKind, ProviderError

__all__ = ["render_prompt"]


def render_prompt(
    template: str,
    context: Mapping[str, Any],
    *,
    node_id: str = "",
    tier: ModelTier | None = None,
) -> str:
    """Render ``template`` against ``context`` via :class:`string.Template`.

    Args:
        template: Source string with ``${name}`` placeholders.
        context: Mapping of name → value substitutions.
        node_id: Optional workflow-node identifier propagated into
            :class:`ProviderError` on failure.
        tier: Optional :class:`ModelTier` propagated into
            :class:`ProviderError` on failure. Defaults to
            :attr:`ModelTier.DEFAULT` so the error is well-formed even
            when called outside a tier context (e.g. unit tests).

    Returns:
        The substituted string.

    Raises:
        ProviderError: When ``context`` is missing a key the template
            references (or the template itself is malformed).
            ``ErrorKind.validation``.
    """
    try:
        return Template(template).substitute(context)
    except (KeyError, ValueError) as exc:
        effective_tier = tier if tier is not None else ModelTier.DEFAULT
        raise ProviderError(
            ErrorKind.validation,
            node_id=node_id,
            tier=effective_tier,
            cause=exc,
            message=f"render_prompt: template substitution failed ({exc!s})",
        ) from exc
