"""Human-readable rendering for validation failures.

Two pure helpers shared by the validators and the ``Validate*`` stages:

- :func:`format_candidates` — render the *valid* alternatives inside a
  :class:`ValidationViolation` message (sorted, comma-joined, capped so a
  huge IR never floods one line).
- :func:`render_violations` — lift a failed report's violations into the
  error message a :class:`StagePersistedFailureError` carries: one
  ``- code: message`` line per unique error violation, so the CLI user sees
  *what* is wrong (names + candidates), never a bare list of codes or a
  JSON dump.

Both are deterministic, no I/O, and never raise — same contract as the
validators that use them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from molexp.harness.schemas.validation import ValidationViolation

__all__ = ["MAX_CANDIDATES", "format_candidates", "render_violations"]

#: Cap on how many candidate names one violation message spells out.
MAX_CANDIDATES = 15


def format_candidates(candidates: Iterable[str], *, limit: int = MAX_CANDIDATES) -> str:
    """Render ``candidates`` as a sorted, comma-joined list capped at ``limit``.

    An empty iterable renders as ``"(none)"``; an over-long list keeps the
    first ``limit`` sorted names and appends ``", ... and N more"``.
    """
    names = sorted(candidates)
    if not names:
        return "(none)"
    if len(names) <= limit:
        return ", ".join(names)
    shown = ", ".join(names[:limit])
    return f"{shown}, ... and {len(names) - limit} more"


def render_violations(violations: Sequence[ValidationViolation]) -> str:
    """One ``- code: message`` line per unique error-severity violation.

    Duplicate ``(code, message)`` pairs (e.g. the same violation fired for
    every member of a bundle) render once. Warnings are omitted — they never
    fail a report, so they don't belong in a failure message.
    """
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for violation in violations:
        if violation.severity != "error":
            continue
        key = (violation.code, violation.message)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {violation.code}: {violation.message}")
    if not lines:
        return "- (no error-severity violations recorded)"
    return "\n".join(lines)
