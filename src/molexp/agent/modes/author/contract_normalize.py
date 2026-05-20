"""Deterministic :class:`~molexp.workflow.WorkflowContract` normalizer.

An LLM-drafted (or a freshly-lowered) :class:`WorkflowContract` often
carries imperfect wiring: an input's ``source`` written as the
``"task_id.output_name"`` long form, an input whose ``source`` is
omitted entirely, or an artifact whose ``produced_by`` does not match
its owning task. :func:`normalize_contract` repairs the mechanically
fixable cases and reports what it could not.

Three repairs, all deterministic — no LLM:

1. **Coerce ``source`` to a bare ``task_id``.** ``"parse.mol"`` becomes
   ``"parse"`` — the contract layer's ``source`` field is documented as
   *the bare task_id*, never ``task_id.output_name``.
2. **Infer a missing ``source``.** When an input has ``source=None``
   and exactly one upstream task declares an output of the same
   ``name``, that task becomes the source.
3. **Derive dependencies.** :func:`derive_dependencies` returns, for
   every task, the ordered tuple of distinct upstream ``task_id``s its
   inputs source from — the ``depends_on`` list a generated
   ``workflow.py`` needs.

Residual issues (an input whose source cannot be inferred, a
``produced_by`` that resolves to no task) are returned as a
:class:`ContractNormalizeReport` rather than silently dropped.

Pure data + pure functions; no LLM, no I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from molexp.workflow import ArtifactDecl, TaskInputSpec, TaskIO, WorkflowContract

__all__ = [
    "ContractNormalizeReport",
    "NormalizeIssue",
    "derive_dependencies",
    "normalize_contract",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class NormalizeIssue(BaseModel):
    """One residual wiring problem the normalizer could not fix.

    Attributes:
        kind: ``unresolved_source`` (an input with no inferable source)
            or ``dangling_produced_by`` (an artifact whose
            ``produced_by`` resolves to no task).
        task_id: The task the issue is attached to.
        detail: A human-readable description.
    """

    model_config = _FROZEN

    kind: str
    task_id: str
    detail: str


class ContractNormalizeReport(BaseModel):
    """The outcome of one :func:`normalize_contract` call.

    Attributes:
        contract: The repaired :class:`WorkflowContract`.
        issues: Residual problems the normalizer could not fix.
    """

    model_config = _FROZEN

    contract: WorkflowContract
    issues: tuple[NormalizeIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the normalized contract has no residual issues."""
        return not self.issues


def _bare_source(source: str | None) -> str | None:
    """Coerce a possibly long-form ``source`` to a bare ``task_id``."""
    if source is None:
        return None
    return source.split(".", 1)[0]


def _outputs_index(contract: WorkflowContract) -> dict[str, list[str]]:
    """Map each output ``name`` to the task ids that declare it."""
    index: dict[str, list[str]] = {}
    for tio in contract.task_io:
        for out in tio.outputs:
            index.setdefault(out.name, []).append(tio.task_id)
    return index


def _infer_source(
    inp: TaskInputSpec,
    *,
    owner: str,
    outputs_index: dict[str, list[str]],
    known_ids: set[str],
) -> tuple[str | None, NormalizeIssue | None]:
    """Resolve one input's source; return (source, residual_issue)."""
    coerced = _bare_source(inp.source)
    if coerced is not None:
        if coerced in known_ids:
            return coerced, None
        return None, NormalizeIssue(
            kind="unresolved_source",
            task_id=owner,
            detail=f"input {inp.name!r} sources from {coerced!r}, which is not a task",
        )
    producers = outputs_index.get(inp.name, [])
    if len(producers) == 1 and producers[0] != owner:
        return producers[0], None
    if not inp.required:
        return None, None
    return None, NormalizeIssue(
        kind="unresolved_source",
        task_id=owner,
        detail=(
            f"input {inp.name!r} has no source and no unique upstream task "
            f"declares an output named {inp.name!r}"
        ),
    )


def normalize_contract(contract: WorkflowContract) -> ContractNormalizeReport:
    """Return a deterministically-repaired copy of ``contract``.

    Coerces every input ``source`` to a bare ``task_id``, infers a
    missing ``source`` from a unique matching upstream output, and
    surfaces the residual issues it could not fix. The input contract
    is never mutated.
    """
    known_ids = {tio.task_id for tio in contract.task_io}
    outputs_index = _outputs_index(contract)
    issues: list[NormalizeIssue] = []
    new_task_io: list[TaskIO] = []

    for tio in contract.task_io:
        new_inputs: list[TaskInputSpec] = []
        for inp in tio.inputs:
            source, issue = _infer_source(
                inp,
                owner=tio.task_id,
                outputs_index=outputs_index,
                known_ids=known_ids,
            )
            if issue is not None:
                issues.append(issue)
            new_inputs.append(inp.model_copy(update={"source": source}))
        new_artifacts: list[ArtifactDecl] = []
        for art in tio.artifacts:
            produced_by = art.produced_by or tio.task_id
            if produced_by not in known_ids:
                issues.append(
                    NormalizeIssue(
                        kind="dangling_produced_by",
                        task_id=tio.task_id,
                        detail=(
                            f"artifact {art.path!r} declares produced_by="
                            f"{art.produced_by!r}, which is not a task"
                        ),
                    )
                )
                produced_by = tio.task_id
            new_artifacts.append(art.model_copy(update={"produced_by": produced_by}))
        new_task_io.append(
            tio.model_copy(update={"inputs": tuple(new_inputs), "artifacts": tuple(new_artifacts)})
        )

    repaired = contract.model_copy(update={"task_io": tuple(new_task_io)})
    return ContractNormalizeReport(contract=repaired, issues=tuple(issues))


def derive_dependencies(contract: WorkflowContract) -> dict[str, tuple[str, ...]]:
    """Return, per task, the ordered tuple of distinct upstream task ids.

    A task depends on every distinct task its inputs source from; the
    order follows first appearance in the input list. The result is the
    ``depends_on`` map a generated ``workflow.py`` needs.
    """
    deps: dict[str, tuple[str, ...]] = {}
    for tio in contract.task_io:
        ordered: list[str] = []
        seen: set[str] = set()
        for inp in tio.inputs:
            source = _bare_source(inp.source)
            if source is not None and source != tio.task_id and source not in seen:
                ordered.append(source)
                seen.add(source)
        deps[tio.task_id] = tuple(ordered)
    return deps
