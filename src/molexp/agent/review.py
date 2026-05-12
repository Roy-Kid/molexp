"""Workflow-orthogonal review primitives for the agent layer.

A single :class:`ReviewPolicy` protocol covers both per-step checkpoints
(inside a multi-step workflow like PlanMode's 13-node pipeline) and
plan-level final approval (the terminal review gate). Both hooks share
the same :class:`ReviewDecision` payload — ``approved`` plus structured
``target_steps`` / ``target_task_ids`` / ``cascade_downstream`` /
``feedback`` fields the repair driver inspects to decide what to re-run.

Three built-in policies cover the common cases:

* :class:`BypassPolicy` — always approves; never blocks.
* :class:`AutoPolicy` — programmatic decision (rules table or LLM
  callback). Used when "the machine decides" should be the default.
* :class:`HumanPolicy` — yields the decision to a human; the rendering
  surface (CLI prompt, web modal, Slack interactive message, mobile
  push) is injected via the ``ask`` callable.  :func:`cli_ask` is the
  bundled CLI implementation; replace it to drive the same policy from
  a different transport without touching ``HumanPolicy`` itself.

The :class:`ReviewView` Protocol describes the minimal payload every
view must surface — concrete shapes (per-step :class:`StepView` and the
mode-specific plan-final view) layer extra fields on top.  Policies that
need to walk those extras switch on ``isinstance`` against the concrete
class; :func:`cli_ask` does exactly that to render a richer prompt for
the plan-final view.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    pass

__all__ = [
    "AutoPolicy",
    "BypassPolicy",
    "HumanPolicy",
    "ReviewDecision",
    "ReviewPolicy",
    "ReviewView",
    "StepView",
    "cli_ask",
]


_FROZEN = ConfigDict(frozen=True, extra="forbid")


# ── Decision ───────────────────────────────────────────────────────────────


class ReviewDecision(BaseModel):
    """Outcome of a single :class:`ReviewPolicy` consultation.

    The same payload shape is returned by every policy at every hook —
    step-level and plan-final.  Field semantics:

    Attributes:
        approved: ``True`` to let the workflow continue past this hook;
            ``False`` to request a repair.  Step-level rejection raises
            a :class:`StepRejected` exception caught by the repair loop;
            plan-final rejection is consumed by the loop directly.
        feedback: Free-form natural-language note persisted to
            ``repairs/iter-<n>/decision.yaml`` and (in future) threaded
            into the next attempt's LLM prompt.
        target_steps: Plan-level node names the repair loop should
            re-execute.  Empty tuple means "the rejected step itself" at
            step-level, or "let the driver pick a default" at
            plan-final level (today: codegen pair).
        target_task_ids: Experiment-task ids whose generated test /
            implementation modules need a fresh LLM pass (consumed by
            ``GenerateTaskTests`` / ``GenerateTaskImplementations``).
            Step-level rejections leave this empty.
        cascade_downstream: When ``True``, every downstream node of the
            ``target_steps`` selection is added to the partial-rerun
            subgraph.
        reason: One-sentence justification, surfaced to logs and audit
            trail.
        override_validation: When ``True``, accepts a plan whose
            validation pass failed; the manifest is then marked
            ``approved_with_override``.  Honoured only by the plan-final
            hook (HumanReview); step-level policies ignore it.
    """

    model_config = _FROZEN

    approved: bool
    feedback: str = ""
    target_steps: tuple[str, ...] = ()
    target_task_ids: tuple[str, ...] = ()
    cascade_downstream: bool = False
    reason: str = ""
    override_validation: bool = False


# ── View Protocols ────────────────────────────────────────────────────────


@runtime_checkable
class ReviewView(Protocol):
    """Minimal contract every view passed to a policy must satisfy.

    Concrete views (:class:`StepView`, PlanMode's ``PlanReviewView`` /
    ``FinalView``) implement this Protocol and may layer additional
    fields; policies that need the extras switch on ``isinstance``.
    """

    @property
    def step_id(self) -> str:
        """Identifier for this checkpoint.  Per-step views return the
        plan node name; plan-final views return the literal ``"final"``."""
        ...

    @property
    def summary(self) -> str:
        """Single-line description of what just completed."""
        ...

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        """Materialized files the reviewer may want to inspect."""
        ...


class StepView(BaseModel):
    """Per-step view passed to :meth:`ReviewPolicy.review` after each node.

    The PlanTask base class constructs one of these once a node's
    ``_execute()`` returns, before deciding whether to surface the
    result to the user.

    Attributes:
        plan_id: Identifier for the plan the step belongs to.
        step_id: Node name (the ``type(self).__name__`` of the Task).
        summary: One-line human-readable note ("CompileWorkflowIR
            produced 5 tasks").
        artifact_paths: Files the node materialized this round.
        payload: ``result.model_dump()`` for inspection — policies that
            want richer context (e.g. a web UI showing the structured
            output) can read this without re-loading from disk.
        repair_iteration: Zero on the first pass; bumped by the repair
            loop on each retry.
    """

    model_config = _FROZEN

    plan_id: str
    step_id: str
    summary: str
    artifact_paths: tuple[Path, ...] = ()
    payload: dict[str, Any] | None = None
    repair_iteration: int = 0


# ── Policy Protocol ────────────────────────────────────────────────────────


@runtime_checkable
class ReviewPolicy(Protocol):
    """Decide whether a workflow checkpoint passes.

    A single ``review`` method matches both step-level and plan-final
    hooks — concrete policies decide what to do with the view they
    receive.  Returning ``ReviewDecision(approved=True)`` lets the
    workflow continue past the checkpoint; ``approved=False`` triggers
    a repair iteration.
    """

    async def review(self, view: ReviewView) -> ReviewDecision: ...


# ── Concrete policies ─────────────────────────────────────────────────────


class BypassPolicy:
    """Approve every checkpoint without inspection.

    The deliberately blunt name signals that this policy makes no
    decision — it just rubber-stamps.  Use as the default for hooks you
    have not configured yet, or in non-interactive scripts where you
    have no human reviewer wired up.  Pair with :class:`HumanPolicy` on
    the hook that *should* prompt.
    """

    def __init__(self, *, reason: str = "bypass") -> None:
        self._reason = reason

    async def review(self, view: ReviewView) -> ReviewDecision:
        del view  # ignored — bypass approves without inspection
        return ReviewDecision(approved=True, reason=self._reason)


class AutoPolicy:
    """Decide programmatically — rule table, LLM call, or any callback.

    The ``decide`` callable receives the view and returns a
    :class:`ReviewDecision`.  Use this when "the machine decides" is the
    right default — e.g. auto-approve safe steps but flag risky ones
    for a human follow-up.

    Example::

        def static_rules(view: ReviewView) -> ReviewDecision:
            risky = {"DiscoverCapabilities", "GenerateTaskImplementations"}
            return ReviewDecision(approved=view.step_id not in risky)


        AutoPolicy(decide=static_rules)
    """

    def __init__(
        self,
        decide: Callable[[ReviewView], ReviewDecision | Awaitable[ReviewDecision]],
    ) -> None:
        self._decide = decide

    async def review(self, view: ReviewView) -> ReviewDecision:
        outcome = self._decide(view)
        if isinstance(outcome, ReviewDecision):
            return outcome
        return await outcome


class HumanPolicy:
    """Yield the decision to a human; rendering is the ``ask`` callable's job.

    The policy itself stays UI-agnostic — :func:`cli_ask` is the
    bundled CLI implementation; web UIs, Slack apps, or mobile push
    flows can plug in their own ``ask`` without touching this class.

    The ``ask`` signature is ``Callable[[ReviewView], Awaitable[ReviewDecision]]``;
    that means: take a view, eventually produce a decision.  How long
    it takes (a synchronous CLI prompt vs a websocket round-trip vs an
    email-and-wait flow) is the callback's concern.
    """

    def __init__(
        self,
        ask: Callable[[ReviewView], Awaitable[ReviewDecision]] | None = None,
    ) -> None:
        self._ask = ask if ask is not None else cli_ask

    async def review(self, view: ReviewView) -> ReviewDecision:
        return await self._ask(view)


# ── CLI rendering ─────────────────────────────────────────────────────────


async def cli_ask(view: ReviewView) -> ReviewDecision:
    """Bundled CLI prompt — the default ``HumanPolicy(ask=...)`` callback.

    Renders a different prompt depending on the concrete view type:

    * Per-step (:class:`StepView` or any other view that is not a
      mode-specific final view): prints the step id, summary, and
      artifact paths; asks ``y / n / ?`` for the whole step.  ``?``
      shows the artifact files inline.
    * Plan-final (currently :class:`~molexp.agent.modes.plan.schemas.PlanReviewView`):
      walks task-by-task through ``view.contract.task_io``, surfacing
      the generated test + implementation files for each.  Per-task
      rejections collect into ``target_task_ids`` and ``feedback``.

    The detection is structural — anything quacking like
    ``view.contract.task_io`` triggers the plan-final flow, so future
    custom plan-final views composed by other modes work without
    changes here.
    """
    # Plan-final flow: surface every task one-by-one.
    final_renderable = _try_render_final_view(view)
    if final_renderable is not None:
        return await final_renderable

    return _render_step_view(view)


def _try_render_final_view(view: ReviewView) -> Awaitable[ReviewDecision] | None:
    """Detect a plan-final view (``view.contract.task_io`` present) and
    return a coroutine that renders the task-by-task prompt.

    Returns ``None`` for ordinary step views — the caller then falls
    through to :func:`_render_step_view`.  Structural detection rather
    than ``isinstance`` so the helper does not import the
    ``modes/plan/schemas`` module at import time.
    """
    contract = getattr(view, "contract", None)
    task_io = getattr(contract, "task_io", None) if contract is not None else None
    if task_io is None:
        return None
    return _render_final_view(view, task_io)


async def _render_final_view(view: ReviewView, task_io: tuple[Any, ...]) -> ReviewDecision:
    """Walk every task in ``task_io`` and collect per-task verdicts."""
    plan_root = Path(getattr(view, "experiment_workspace_path", Path()))
    tests_dir = plan_root / "tests"
    impls_dir = plan_root / "src" / "experiment" / "tasks"

    print()
    print("=" * 72)
    iteration = getattr(view, "repair_iteration", 0)
    print(f"Review checkpoint — iteration {iteration}")
    print("=" * 72)
    plan_id = getattr(view, "plan_id", "")
    if plan_id:
        print(f"plan_id            : {plan_id}")
    print(f"workspace          : {plan_root}")
    validation_passed = getattr(view, "validation_passed", None)
    if validation_passed is not None:
        print(f"validation passed  : {validation_passed}")
    validation_summary = getattr(view, "validation_summary", None)
    if validation_summary:
        print(f"validation summary : {validation_summary}")
    failures = getattr(view, "previous_validation_failures", ())
    if failures:
        print("failed checks      :")
        for name in failures:
            print(f"  - {name}")

    digest = getattr(view, "digest", None)
    if digest is not None and getattr(digest, "experimental_goal", None):
        print()
        print(f"experimental goal  : {digest.experimental_goal}")
    brief = getattr(view, "plan_brief", None)
    if brief is not None and getattr(brief, "overview", None):
        print(f"plan overview      : {brief.overview}")
    if brief is not None and getattr(brief, "stages", ()):
        print("plan stages        :")
        for stage in brief.stages:
            print(f"  - {stage}")

    if not task_io:
        verdict = input("\nNo tasks in the contract. approve plan? [y/N]: ").strip().lower()
        return ReviewDecision(
            approved=verdict in ("y", "yes"),
            reason="empty contract",
        )

    print()
    print(
        f"{len(task_io)} task(s) to review one-by-one. y=approve, n=reject (replan), ?=show files."
    )

    rejected: list[str] = []
    feedback_lines: list[str] = []
    for idx, entry in enumerate(task_io, 1):
        task_id = getattr(entry, "task_id", str(idx))
        test_path = tests_dir / f"test_{task_id}.py"
        impl_path = impls_dir / f"{task_id}.py"
        print()
        print("-" * 72)
        print(f"Task {idx}/{len(task_io)}: {task_id}")
        print("-" * 72)
        print(f"  test : {test_path}")
        print(f"  impl : {impl_path}")
        choice = _prompt_choice(
            "    approve this task? [y/N/?]: ",
            allow_show=True,
        )
        if choice == "?":
            _print_file(test_path, label="TEST")
            _print_file(impl_path, label="IMPL")
            choice = _prompt_choice(
                "    approve this task? [y/N]: ",
                allow_show=False,
            )
        if choice == "y":
            continue
        note = input("    what should the next round fix? (one line): ").strip()
        rejected.append(task_id)
        if note:
            feedback_lines.append(f"{task_id}: {note}")

    if not rejected:
        return ReviewDecision(
            approved=True,
            reason=f"all {len(task_io)} task(s) approved",
        )
    return ReviewDecision(
        approved=False,
        reason=f"{len(rejected)}/{len(task_io)} task(s) rejected",
        target_task_ids=tuple(rejected),
        cascade_downstream=True,
        feedback="\n".join(feedback_lines),
    )


def _render_step_view(view: ReviewView) -> ReviewDecision:
    """Render a per-step prompt — step id, summary, artifact paths, y/n/?."""
    step_id = getattr(view, "step_id", "?")
    summary = getattr(view, "summary", "")
    artifact_paths = tuple(getattr(view, "artifact_paths", ()))

    print()
    print("-" * 72)
    print(f"Step complete: {step_id}")
    print("-" * 72)
    if summary:
        print(f"  summary: {summary}")
    if artifact_paths:
        print("  artifacts:")
        for path in artifact_paths:
            print(f"    {path}")
    choice = _prompt_choice(
        f"  approve {step_id}? [y/N/?]: ",
        allow_show=bool(artifact_paths),
    )
    if choice == "?":
        for path in artifact_paths:
            _print_file(path, label=path.name)
        choice = _prompt_choice(
            f"  approve {step_id}? [y/N]: ",
            allow_show=False,
        )
    if choice == "y":
        return ReviewDecision(approved=True, reason=f"step {step_id} approved")
    note = input("    what should the next attempt fix? (one line): ").strip()
    return ReviewDecision(
        approved=False,
        reason=f"step {step_id} rejected",
        target_steps=(step_id,),
        cascade_downstream=True,
        feedback=note,
    )


def _prompt_choice(prompt: str, *, allow_show: bool) -> str:
    """Read one of {y, n, ?}; return the canonical letter (default ``n``)."""
    raw = input(prompt).strip().lower()
    if raw in ("y", "yes"):
        return "y"
    if allow_show and raw == "?":
        return "?"
    return "n"


def _print_file(path: Path, *, label: str) -> None:
    """Dump ``path`` to stdout with a banner; tolerate missing files."""
    print(f"  --- {label}: {path} ---")
    if not path.exists():
        print(f"  (missing: {path})")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"  (read failed: {exc})")
        return
    for line in text.splitlines():
        print(f"  | {line}")
    print(f"  --- end {label} ---")
