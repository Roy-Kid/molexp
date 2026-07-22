"""SDK-free tool/hook vocabulary — ``molexp.agent.loops.hooks``.

The domain-neutral primitives a tool-using agent loop consults at three
boundaries: *before* a tool call, *after* a tool result, and *before*
declaring the loop finished. Every primitive here is stdlib + pydantic
only — no ``pydantic_ai`` import, no harness type (``ToolCapability`` /
``ApprovalPendingError``) — so the vocabulary stays a pure contract that
both the agent loop *and* a harness-injected gate can speak without either
side dragging the other's dependencies in.

The vocabulary is deliberately **domain-neutral**: a ``BeforeToolHook`` knows
only a tool name and its opaque args, never *why* a caller might veto it. In
phase 02 the emergent loop (:class:`~molexp.agent.loops.InteractiveLoop`) and
harness-injected approval gates plug concrete policies into these Protocols;
phase 01 only pins the shapes and the honor semantics of the ``invoke_*``
helpers.

Modeled on :mod:`molexp.agent.router` — frozen :class:`pydantic.BaseModel`
data, a :class:`enum.StrEnum` decision axis, and ``@runtime_checkable``
:class:`typing.Protocol` callables.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

__all__ = [
    "AfterToolHook",
    "BeforeToolHook",
    "HookDecision",
    "HookOutcome",
    "LoopHooks",
    "LoopState",
    "ShouldStopGuard",
    "invoke_after_tool",
    "invoke_before_tool",
    "invoke_should_stop",
]


# ── Decision axis ──────────────────────────────────────────────────────────


class HookDecision(StrEnum):
    """What a hook asks the loop to do at its boundary.

    ``PROCEED`` is the always-safe default (and what a missing hook implies);
    ``DENY`` vetoes the pending action; ``SUSPEND`` parks the loop pending an
    out-of-band resolution keyed by a token. Enforcement of ``DENY`` /
    ``SUSPEND`` is realized in phase 02 — phase 01 only carries the decision.
    """

    PROCEED = "proceed"
    DENY = "deny"
    SUSPEND = "suspend"


# ── Hook outcome ───────────────────────────────────────────────────────────


class HookOutcome(BaseModel):
    """One hook's verdict — a :class:`HookDecision` plus optional payload.

    Frozen. Construct through the :meth:`proceed` / :meth:`deny` /
    :meth:`suspend` factories rather than the raw constructor so the payload
    field carried alongside each decision stays consistent (``message`` for a
    deny, ``token`` for a suspend, neither for a proceed).

    Attributes:
        decision: The verdict this outcome carries.
        message: Human-readable rationale — set on a ``DENY``, empty otherwise.
        token: Correlation token for out-of-band resolution — set on a
            ``SUSPEND``, empty otherwise.
    """

    model_config = ConfigDict(frozen=True)

    decision: HookDecision
    message: str = ""
    token: str = ""

    @classmethod
    def proceed(cls) -> HookOutcome:
        """Return the always-safe ``PROCEED`` outcome (no payload)."""
        return cls(decision=HookDecision.PROCEED)

    @classmethod
    def deny(cls, message: str) -> HookOutcome:
        """Return a ``DENY`` outcome carrying ``message`` as the rationale."""
        return cls(decision=HookDecision.DENY, message=message)

    @classmethod
    def suspend(cls, token: str) -> HookOutcome:
        """Return a ``SUSPEND`` outcome carrying ``token`` for later resume."""
        return cls(decision=HookDecision.SUSPEND, token=token)

    @property
    def is_proceed(self) -> bool:
        """Whether the decision is :attr:`HookDecision.PROCEED`."""
        return self.decision == HookDecision.PROCEED

    @property
    def is_deny(self) -> bool:
        """Whether the decision is :attr:`HookDecision.DENY`."""
        return self.decision == HookDecision.DENY

    @property
    def is_suspend(self) -> bool:
        """Whether the decision is :attr:`HookDecision.SUSPEND`."""
        return self.decision == HookDecision.SUSPEND


# ── Loop state snapshot ────────────────────────────────────────────────────


class LoopState(BaseModel):
    """Minimal, SDK-free snapshot a :class:`ShouldStopGuard` reasons over.

    Frozen. Carries only what a stop guard needs — the loop step count, the
    most recently dispatched tool, and the draft final answer assembled so far.

    Attributes:
        step: 1-based count of loop steps taken.
        last_tool: Name of the most recent tool dispatched, or ``""`` if none.
        draft_final: The draft final answer assembled so far, or ``""``.
    """

    model_config = ConfigDict(frozen=True)

    step: int
    last_tool: str = ""
    draft_final: str = ""


# ── Hook Protocols ─────────────────────────────────────────────────────────


@runtime_checkable
class BeforeToolHook(Protocol):
    """Consulted *before* a tool call is dispatched.

    A conforming object is any async callable taking the pending tool's name
    and its opaque args and returning a :class:`HookOutcome`.
    """

    async def __call__(
        self, *, tool_name: str, args: Mapping[str, Any] | str | None
    ) -> HookOutcome:
        """Return the outcome for the pending call to ``tool_name``."""
        ...


@runtime_checkable
class AfterToolHook(Protocol):
    """Consulted *after* a tool result is observed.

    A conforming object is any async callable taking the tool's name and a
    short result summary and returning a :class:`HookOutcome`.
    """

    async def __call__(self, *, tool_name: str, result: str) -> HookOutcome:
        """Return the outcome for the observed ``result`` of ``tool_name``."""
        ...


@runtime_checkable
class ShouldStopGuard(Protocol):
    """Consulted *before* the loop declares itself finished.

    A conforming object is any async callable taking a :class:`LoopState`
    snapshot and returning a :class:`HookOutcome`.
    """

    async def __call__(self, *, state: LoopState) -> HookOutcome:
        """Return the outcome for stopping the loop in ``state``."""
        ...


# ── Hook bundle ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LoopHooks:
    """The three-boundary hook bundle a loop is constructed with.

    A plain frozen :func:`dataclasses.dataclass` — **not** a pydantic model —
    because it carries live async callables (the hook Protocols), which the
    agent-layer charter forbids inside a frozen ``BaseModel``
    (``arbitrary_types_allowed`` is banned in ``molexp.agent``). Every field
    defaults to ``None`` (opt out), so an all-``None`` bundle is behaviorally
    equivalent to passing no hooks at all.

    Attributes:
        before_tool: Consulted before each tool call, or ``None`` to opt out.
        after_tool: Consulted after each tool result, or ``None`` to opt out.
        should_stop: Consulted before the loop declares itself finished, or
            ``None`` to opt out.
    """

    before_tool: BeforeToolHook | None = None
    after_tool: AfterToolHook | None = None
    should_stop: ShouldStopGuard | None = None


# ── Honor helpers (single source of the None → proceed semantics) ──────────


async def invoke_before_tool(
    hook: BeforeToolHook | None,
    *,
    tool_name: str,
    args: Mapping[str, Any] | str | None,
) -> HookOutcome:
    """Invoke ``hook`` for a pending tool call, honoring ``None`` as proceed.

    Args:
        hook: The before-tool hook, or ``None`` to opt out.
        tool_name: Name of the pending tool call.
        args: The call's opaque arguments (mapping / string / ``None``).

    Returns:
        :meth:`HookOutcome.proceed` when ``hook`` is ``None``; otherwise the
        hook's own outcome, verbatim.
    """
    if hook is None:
        return HookOutcome.proceed()
    return await hook(tool_name=tool_name, args=args)


async def invoke_after_tool(
    hook: AfterToolHook | None, *, tool_name: str, result: str
) -> HookOutcome:
    """Invoke ``hook`` for an observed tool result, honoring ``None`` as proceed.

    Args:
        hook: The after-tool hook, or ``None`` to opt out.
        tool_name: Name of the tool whose result was observed.
        result: A short summary of the tool's result.

    Returns:
        :meth:`HookOutcome.proceed` when ``hook`` is ``None``; otherwise the
        hook's own outcome, verbatim.
    """
    if hook is None:
        return HookOutcome.proceed()
    return await hook(tool_name=tool_name, result=result)


async def invoke_should_stop(guard: ShouldStopGuard | None, *, state: LoopState) -> HookOutcome:
    """Invoke ``guard`` at loop end, honoring ``None`` as proceed.

    Args:
        guard: The should-stop guard, or ``None`` to opt out.
        state: The current loop-state snapshot.

    Returns:
        :meth:`HookOutcome.proceed` when ``guard`` is ``None``; otherwise the
        guard's own outcome, verbatim.
    """
    if guard is None:
        return HookOutcome.proceed()
    return await guard(state=state)
