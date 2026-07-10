"""``harvest_run`` — turn a finished Run's outcome into typed knowledge.

The explicit execution→knowledge verb for plain runs (vision-loop-06): a
researcher (or, later, an agent capability) harvests a terminal run into a
source-attributed :class:`KnowledgeItem` mounted under the run's experiment.

Harvesting is **interpretation, not archival**: the run's raw record already
lives in ``run.json``/artifacts, so a harvest without a narrative is refused —
knowledge is what the numbers *mean*. Preconditions error, never fall back:

* the run must be terminal (``succeeded`` / ``failed`` / ``cancelled``) — a
  live run has no outcome to interpret yet;
* ``narrative`` must be non-empty.

Placement rationale: workspace-only imports (Run / KnowledgeItem / Folder), so
the P1 lifecycle-capability slice can expose this as a harness
``ToolCapability`` (harness→workspace is legal; harness→services is not).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.ids import slugify

from .knowledge_item import KnowledgeItem, KnowledgeKind, SourceRef
from .knowledge_write import write_knowledge_item

if TYPE_CHECKING:
    from molexp._typing import JSONValue

    from .run import Run

__all__ = ["harvest_run"]

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

_MAX_VALUE_CHARS = 400
"""Per-value cap in the rendered results table (repr-truncated)."""


def harvest_run(
    run: Run,
    *,
    kind: KnowledgeKind,
    narrative: str,
    created_by: str,
    results: dict[str, JSONValue] | None = None,
    name: str | None = None,
) -> KnowledgeItem:
    """Harvest a terminal *run* into a typed, source-attributed KnowledgeItem.

    Args:
        run: The finished run whose outcome is being interpreted.
        kind: The knowledge category (``"Finding"``, ``"FailureAnalysis"``, …).
        narrative: The human/agent interpretation — required, non-empty.
        created_by: Who harvested (a person, or ``"agent:<name>"``).
        results: Optional key→value table of the outcome's headline numbers;
            values are repr-truncated at ``_MAX_VALUE_CHARS``.
        name: Explicit Concept name for multiple harvests of one run; the
            default ``f"{slug(kind)}-{run.id}"`` makes re-harvesting
            idempotent (same name → update in place).

    Returns:
        The written :class:`KnowledgeItem`, mounted under the run's experiment.

    Raises:
        ValueError: The run is not terminal, or *narrative* is empty.
    """
    if run.status not in _TERMINAL_STATUSES:
        raise ValueError(
            f"run {run.id} is {run.status!r} — only a terminal run "
            f"({sorted(_TERMINAL_STATUSES)}) has an outcome to harvest"
        )
    if not narrative.strip():
        raise ValueError(
            "harvest requires a non-empty narrative — knowledge is interpretation; "
            "the raw record already lives in run.json and the run's artifacts"
        )

    experiment = run.experiment
    item_name = name or f"{slugify(kind)}-{run.id}"
    return write_knowledge_item(
        experiment,
        name=item_name,
        kind=kind,
        sources=[
            SourceRef(kind="run", ref=run.id),
            SourceRef(kind="experiment", ref=experiment.id),
        ],
        created_by=created_by,
        body=_render_body(run, kind=kind, narrative=narrative, results=results),
        cite=[(run, "derived_from")],
    )


def _render_body(
    run: Run,
    *,
    kind: KnowledgeKind,
    narrative: str,
    results: dict[str, JSONValue] | None,
) -> str:
    lines = [
        f"# [{kind}] run {run.id}",
        "",
        narrative.strip(),
        "",
        "## Run",
        "",
        f"- status: {run.status}",
    ]
    params = run.metadata.parameters
    if params:
        lines.append(f"- params: {_truncate(repr(params))}")
    error = run.metadata.error
    if run.status == "failed" and error:
        lines += ["", "## Error", "", _truncate(str(error))]
    if results:
        lines += ["", "## Results", ""]
        lines += [f"- **{key}**: {_truncate(repr(value))}" for key, value in results.items()]
    return "\n".join(lines).rstrip() + "\n"


def _truncate(text: str) -> str:
    if len(text) <= _MAX_VALUE_CHARS:
        return text
    return text[:_MAX_VALUE_CHARS] + f"… (+{len(text) - _MAX_VALUE_CHARS} chars)"
