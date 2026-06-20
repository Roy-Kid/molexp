"""Workspace utility functions.

The layer-agnostic id / slug / content-hash primitives (``slugify``,
``generate_id``, ``generate_asset_id``, ``compute_content_hash``) moved to
the cross-layer primitive :mod:`molexp.ids` (okf-01-01) so the OKF
``knowledge`` bottom layer can cite them without importing workspace. They
remain importable from ``molexp.workspace.utils`` (same function objects)
for back-compat. The run-domain id derivations below
(``derive_run_id`` / ``derive_execution_id``) stay here — they are
workspace-specific, not layer-agnostic primitives.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from molexp.ids import (
    compute_content_hash,
    generate_asset_id,
    generate_id,
    slugify,
)

if TYPE_CHECKING:
    from .._typing import JSONValue

__all__ = [
    "compute_content_hash",
    "derive_execution_id",
    "derive_run_id",
    "generate_asset_id",
    "generate_id",
    "slugify",
]


def derive_run_id(params: Mapping[str, JSONValue], *, length: int = 16) -> str:
    """Derive a deterministic, content-addressed run id from a parameter dict.

    The id is a sha256 over the canonicalized parameters — keys sorted, each
    rendered ``k=repr(v)`` — so it is a pure function of the params and
    independent of dict insertion order. Identical params always map to the
    same id, which makes run materialization idempotent (see
    :meth:`Experiment.add_runs`). This is the single canonicalization shared by
    the workspace layer and ``cli._common.deterministic_run_id``.

    Args:
        params: The run's parameter mapping (JSON-serializable values).
        length: Number of leading hex characters to keep (default 16).

    Returns:
        A ``length``-character lowercase hex string.
    """
    raw = "|".join(f"{k}={v!r}" for k, v in sorted(params.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:length]


def derive_execution_id(run_id: str, exec_root: Path) -> str:
    """Return the execution_id for the next attempt of *run_id*.

    First attempt is ``exec-{run_id}``; retries are ``exec-{run_id}-2``,
    ``exec-{run_id}-3``, … The next suffix is ``max(existing suffix) + 1``
    (never ``len(existing)``): deleting a *middle* attempt would shrink the
    count and make ``len`` collide with a still-present higher id, whereas
    ``max + 1`` is always strictly greater than every live attempt, so it
    never reuses an existing id.

    Attempts are matched by *exact* name — ``exec-{run_id}`` itself (attempt
    1) or ``exec-{run_id}-<int>`` — so a run whose id is a string prefix of
    another run's id (``"ab"`` vs ``"abc"``) is not miscounted, and non-numeric
    suffixes are ignored.

    This is the single source of execution-id derivation; the workflow runtime
    (:func:`molexp.workflow.make_execution_id`) and the workspace
    ``ExecutionStore`` both delegate here.
    """
    base = f"exec-{run_id}"
    if not exec_root.exists():
        return base
    prefix = f"{base}-"
    highest = 0  # 0 → no attempt yet; `base` itself counts as attempt 1
    for entry in exec_root.iterdir():
        name = entry.name
        if name == base:
            highest = max(highest, 1)
        elif name.startswith(prefix):
            tail = name[len(prefix) :]
            if tail.isdigit():
                highest = max(highest, int(tail))
    if highest == 0:
        return base
    return f"{base}-{highest + 1}"
