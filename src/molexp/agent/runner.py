"""``AgentRunner`` — public orchestration entry point.

Takes a loop + a model config; builds an
:class:`~molexp.agent.runtime.AgentRuntime`, injects it into the
loop, drains the loop's :data:`~molexp.agent.events.AgentEvent`
stream through an :class:`AsyncIteratorEventSink`, and returns the
terminal :class:`~molexp.agent.loop.AgentRunResult`.

The router is constructed lazily on first :meth:`run` — the private
:class:`~molexp.agent._pydanticai.router.PydanticAIRouter` is the only
``pydantic_ai`` construction site and users never see it directly.

Three mutually-exclusive ways to specify the model:

* ``model="deepseek:deepseek-v4-flash"`` — single string, applied to
  every tier (``CHEAP`` / ``DEFAULT`` / ``HEAVY``).
* ``models={ModelTier.CHEAP: ..., ...}`` — explicit per-tier mapping.
  String tier keys (``"cheap"`` etc.) are accepted and coerced.
* ``router=<custom Router>`` — escape hatch for tests and fakes.

Exactly one must be supplied; zero or two-or-more raise
:class:`AgentRunnerConfigError` at construction.

Two run surfaces:

* :meth:`run` — drains the loop's event stream and returns the terminal
  :class:`AgentRunResult` (the back-compat shape, now with ``events``).
* :meth:`run_events` — an async generator exposing the live event
  stream for the future SSE consumer.

Named sessions
==============

When ``workspace=<path>`` is supplied, the runner anchors each
conversation to a :class:`~molexp.agent.folders.AgentSession` ``Folder``
under an :class:`~molexp.agent.folders.Agent` named after the loop, and
backs the :class:`~molexp.agent.session.Session` with a
:class:`~molexp.agent.session_storage.JsonlSessionStorage`
writing ``entries.jsonl`` in that folder's directory. Without a
workspace, sessions use an
:class:`~molexp.agent.session_storage.InMemorySessionStorage`.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger

from molexp.agent.events import AgentEvent, AsyncIteratorEventSink, LoopCompletedEvent
from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.loop import AgentRunResult
from molexp.agent.router import ModelTier, Router, TierModels
from molexp.agent.runtime import AgentRuntime
from molexp.agent.session import Session
from molexp.agent.session_storage import (
    InMemorySessionStorage,
    JsonlSessionStorage,
)

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool

    from molexp.agent.folders import Agent as AgentFolder
    from molexp.agent.loop import AgentLoop


_LOG = get_logger(__name__)


__all__ = ["AgentRunner", "AgentRunnerConfigError"]


class AgentRunnerConfigError(ValueError):
    """Raised when :class:`AgentRunner`'s model configuration is unusable.

    Three failure modes:

    1. Zero of ``model`` / ``models`` / ``router`` supplied.
    2. Two or more supplied (ambiguous).
    3. ``models=`` provided but missing one of the three :class:`ModelTier`
       keys.
    """


class AgentRunner:
    """Drive an ``AgentLoop`` end-to-end with an :class:`AgentRuntime`.

    Construction performs no network IO — the underlying pydantic-ai
    ``Agent``\\ s are built lazily on first :meth:`run`.
    """

    def __init__(
        self,
        *,
        loop: AgentLoop,
        model: str | object | None = None,
        models: Mapping[ModelTier | str, str | object] | None = None,
        router: Router | None = None,
        tools: tuple[Tool[None] | Callable[..., Any], ...] = (),
        workspace: Path | None = None,
    ) -> None:
        supplied = sum(x is not None for x in (model, models, router))
        if supplied == 0:
            raise AgentRunnerConfigError(
                "AgentRunner requires one of: model=<str>, models=<tier→model map>, "
                "or router=<custom Router>."
            )
        if supplied > 1:
            raise AgentRunnerConfigError(
                "AgentRunner accepts exactly one of model=, models=, router=. "
                f"Got {supplied} of them."
            )

        self.loop = loop
        self.tools = tools
        self.workspace = workspace
        self._router: Router | None = router
        self._tier_models: TierModels | None
        if router is not None:
            self._tier_models = None
        elif model is not None:
            self._tier_models = dict.fromkeys(ModelTier, model)
        else:
            assert models is not None  # narrowed by the count check above
            self._tier_models = _normalize_tier_map(models)
        self._agent_folder: AgentFolder | None = None

    @property
    def model(self) -> object | None:
        """Model id (or model object) for the ``DEFAULT`` tier.

        Returns ``None`` when a custom :class:`Router` was injected.
        """
        if self._tier_models is None:
            return None
        return self._tier_models[ModelTier.DEFAULT]

    # ── run surfaces ────────────────────────────────────────────────────────

    async def run(self, session: Session, user_input: str) -> AgentRunResult:
        """Drive the loop and return its terminal :class:`AgentRunResult`.

        Drains the loop's :data:`AgentEvent` stream, accumulates every
        event, and folds the terminal
        :class:`~molexp.agent.events.LoopCompletedEvent` into
        the returned result (whose ``events`` field carries the whole
        stream).
        """
        accumulated: list[AgentEvent] = []
        async for event in self.run_events(session, user_input):
            accumulated.append(event)
        return _result_from_stream(tuple(accumulated))

    async def run_events(self, session: Session, user_input: str) -> AsyncIterator[AgentEvent]:
        """Drive the loop and yield its :data:`AgentEvent` stream live.

        Loops are plain ``async def`` coroutines after spec
        ``harness-as-mode-substrate-03b``: they accept the
        :class:`AgentRuntime` bundle + the sink + the user prompt and
        return ``None``; every event flows through the sink. The runner
        spawns a driver task to run the loop and iterates the sink in
        emission order. When the loop finishes (or raises), the driver
        closes the sink, the consumer's ``async for`` loop terminates
        naturally, and any exception the loop raised is re-raised here.

        Cancellation safety: if the consumer breaks out early the driver
        task is cancelled and awaited before this generator exits, so
        no orphan task is left behind.
        """
        router = self._ensure_router()
        sink = AsyncIteratorEventSink()
        runtime = AgentRuntime(
            session=session,
            router=router,
            execution_env=self._build_execution_env(),
        )

        driver_exc: Exception | None = None

        async def _drive() -> None:
            nonlocal driver_exc
            try:
                await self.loop.run(
                    runtime=runtime,
                    sink=sink,
                    user_input=user_input,
                )
            except Exception as exc:
                driver_exc = exc
            finally:
                await sink.close()

        driver = asyncio.create_task(_drive())
        try:
            async for event in sink:
                yield event
        finally:
            if not driver.done():
                driver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await driver

        if driver_exc is not None:
            raise driver_exc

    # ── named sessions ──────────────────────────────────────────────────────

    def session(self, session_id: str) -> Session:
        """Return a :class:`Session` named ``session_id``.

        With a workspace, the session is backed by a
        :class:`JsonlSessionStorage` anchored to a
        :class:`~molexp.agent.folders.AgentSession` ``Folder`` under an
        :class:`~molexp.agent.folders.Agent` named after the loop —
        ``entries.jsonl`` survives across processes. Without a
        workspace, an :class:`InMemorySessionStorage` is used.
        """
        directory = self._session_directory(session_id)
        if directory is not None:
            return Session(storage=JsonlSessionStorage(directory), session_id=session_id)
        return Session(storage=InMemorySessionStorage(), session_id=session_id)

    def _session_directory(self, session_id: str) -> Path | None:
        """Return the on-disk anchor dir for ``session_id``, or ``None``.

        Mounts (or attaches to) the loop's :class:`Agent` folder and a
        named :class:`~molexp.agent.folders.AgentSession` child, then
        resolves its directory. Returns ``None`` when no workspace is
        configured or the folder cannot be opened.
        """
        agent_folder = self._ensure_agent_folder()
        if agent_folder is None:
            return None
        try:
            if agent_folder.has_session(session_id):
                sess_folder = agent_folder.get_session(session_id)
            else:
                sess_folder = agent_folder.add_session(session_id)
            return Path(str(sess_folder.path()))
        except OSError as exc:  # pragma: no cover — read-only fs / schema drift
            _LOG.warning(
                f"[runner] session({session_id!r}): could not open on-disk "
                f"anchor ({exc!r}); using in-memory storage."
            )
            return None

    def _ensure_agent_folder(self) -> AgentFolder | None:
        """Lazily mount the persistent :class:`Agent` folder for this runner."""
        if self._agent_folder is not None:
            return self._agent_folder
        if self.workspace is None:
            return None
        try:
            from molexp.agent.folders import Agent as AgentFolder

            # The agent is a knowledge Concept rooted at the workspace path;
            # construction is I/O-free and idempotent (same path → same dir),
            # and add_session lazily materializes it.
            agent_name = getattr(self.loop, "name", "") or "default"
            self._agent_folder = AgentFolder(name=agent_name, root=Path(self.workspace))
        except OSError as exc:
            _LOG.warning(
                f"[runner] could not open Agent folder for {self.workspace!r}: "
                f"{exc!r}; sessions will be in-memory only."
            )
            return None
        return self._agent_folder

    # ── internals ───────────────────────────────────────────────────────────

    def _ensure_router(self) -> Router:
        """Build the pydantic-ai router lazily on first run."""
        if self._router is not None:
            return self._router
        from molexp.agent._pydanticai.router import PydanticAIRouter

        assert self._tier_models is not None
        preamble = self._compose_system_prompt()
        kwargs: dict[str, Any] = {
            "models": self._tier_models,
            "tools": self.tools,
            "workspace": self.workspace,
        }
        if preamble:
            kwargs["system_prompt"] = preamble
        self._router = PydanticAIRouter(**kwargs)
        return self._router

    def _build_execution_env(self) -> LocalExecutionEnv:
        """Construct the :class:`LocalExecutionEnv` for the harness.

        The scratch dir lives under the workspace when one is configured
        (``<workspace>/.agent-scratch``), otherwise under a process-temp
        directory.
        """
        if self.workspace is not None:
            scratch = Path(self.workspace) / ".agent-scratch"
        else:
            import tempfile

            scratch = Path(tempfile.gettempdir()) / "molexp-agent-scratch"
        return LocalExecutionEnv(scratch_dir=scratch)

    def _compose_system_prompt(self) -> str:
        """Concatenate MCP ``usage_instructions`` + the workspace path note.

        Returns the empty string when there is no preamble *and* no
        workspace to advertise. Construction errors are non-fatal.
        """
        fragments: list[str] = []
        try:
            from molexp.agent.mcp.store import McpStore

            workspace_root = self.workspace if self.workspace is not None else Path()
            store = McpStore(workspace_root)
            entries = store.list()
        except OSError:
            entries = []

        fragments.extend(
            entry.usage_instructions
            for entry in entries
            if entry.valid and not entry.shadowed and entry.usage_instructions
        )
        if self.workspace is not None:
            fragments.append(f"Workspace: {Path(self.workspace).resolve()}")
        return "\n\n".join(fragments)


# ── stream accumulation ────────────────────────────────────────────────────


def _result_from_stream(events: tuple[AgentEvent, ...]) -> AgentRunResult:
    """Fold an accumulated event stream into the terminal :class:`AgentRunResult`.

    The loop's terminal
    :class:`~molexp.agent.events.LoopCompletedEvent` carries the
    result's JSON dump in ``result``; we rebuild the typed result from
    it and attach the whole stream as ``events``.
    """
    terminal: LoopCompletedEvent | None = None
    for event in events:
        if isinstance(event, LoopCompletedEvent):
            terminal = event
    if terminal is None:
        raise RuntimeError(
            "the loop's event stream ended without a LoopCompletedEvent; "
            "every AgentLoop.run must yield one as its terminal event."
        )
    if terminal.result is not None:
        payload = dict(terminal.result)
        payload.pop("events", None)  # rebuilt from the stream below
        base = AgentRunResult.model_validate(payload)
    else:
        base = AgentRunResult(text=terminal.text)
    return base.model_copy(update={"events": events})


def _normalize_tier_map(
    raw: Mapping[ModelTier | str, str | object],
) -> dict[ModelTier, str | object]:
    """Coerce string keys (``"cheap"``) to :class:`ModelTier` and validate
    that every tier is covered."""
    coerced: dict[ModelTier, str | object] = {}
    for raw_key, value in raw.items():
        if isinstance(raw_key, ModelTier):
            tier = raw_key
        elif isinstance(raw_key, str):
            try:
                tier = ModelTier(raw_key)
            except ValueError as exc:
                raise AgentRunnerConfigError(
                    f"AgentRunner.models has unknown tier key {raw_key!r}; "
                    f"must be one of {[t.value for t in ModelTier]}."
                ) from exc
        else:
            raise AgentRunnerConfigError(
                f"AgentRunner.models keys must be ModelTier or str; got {type(raw_key).__name__}."
            )
        coerced[tier] = value
    missing = [tier.value for tier in ModelTier if tier not in coerced]
    if missing:
        raise AgentRunnerConfigError(
            f"AgentRunner.models must cover every ModelTier; missing: {missing}."
        )
    return coerced
