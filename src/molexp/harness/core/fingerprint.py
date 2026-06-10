"""``stage_fingerprint`` — code identity for the Mode completion ledger.

The ledger (see :class:`molexp.harness.mode.Mode`) must not silently reuse a
stage's prior artifact after the stage's *implementation* changed. Each ledger
entry therefore records the producing stage's code fingerprint; on resume a
mismatching fingerprint drops the entry (warn, recompute — never error),
mirroring the workflow layer's seed-validation semantics.

The fingerprint is a sha256 over the stage class's module-qualified name plus
its AST-normalized source (``ast.parse`` → ``ast.unparse``), so whitespace and
comment edits do not invalidate the ledger but any code change does. A class
whose source cannot be retrieved (e.g. defined in a REPL) degrades to the
qualified name alone — weaker, but still stable across runs.

This is deliberately a harness-local helper: the workflow layer's
``TaskSnapshot`` solves the same problem for task bodies, but its
normalization helpers are private to that layer and shaped around callables
with code objects, not ABC subclasses.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.harness.core.stage import Stage

__all__ = ["stage_fingerprint"]


def stage_fingerprint(stage: Stage) -> str:
    """Return the code-identity fingerprint of ``stage``'s class.

    Args:
        stage: The stage instance whose class identity is fingerprinted.
            Instance configuration is *not* part of the fingerprint — the
            ledger already keys on the mode's ``user_input`` and the stage
            ``name``.

    Returns:
        ``"sha256:<hex>"`` over the class's qualified name + normalized
        source (or the qualified name alone when source is unavailable).
    """
    cls = type(stage)
    identity = f"{cls.__module__}.{cls.__qualname__}"
    try:
        source = textwrap.dedent(inspect.getsource(cls))
        normalized = ast.unparse(ast.parse(source))
    except (OSError, TypeError, SyntaxError):
        normalized = ""
    digest = hashlib.sha256(f"{identity}\n{normalized}".encode()).hexdigest()
    return f"sha256:{digest}"
