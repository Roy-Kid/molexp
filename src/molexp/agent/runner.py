"""``AgentRunner`` — one user message, one model call (text or ReAct).

Builds an :class:`~molexp.agent.runtime.AgentRuntime`, drives either
``Router.complete_text`` (``mode="text"``) or one ReAct
(``Router.stream_agentic``, ``mode="agentic"``), and returns the
terminal :class:`~molexp.agent.loop.AgentRunResult`.

The router is constructed lazily on first :meth:`run` — the private
:class:`~molexp.agent._pydanticai.router.PydanticAIRouter` is the only
``pydantic_ai`` construction site.

Three mutually-exclusive ways to specify the model:

* ``model="deepseek:deepseek-v4-flash"`` — single string, applied to
  every tier (``CHEAP`` / ``DEFAULT`` / ``HEAVY``).
* ``models={ModelTier.CHEAP: ..., ...}`` — explicit per-tier mapping.
* ``router=<custom Router>`` — escape hatch for tests and fakes.

Exactly one must be supplied; zero or two-or-more raise
:class:`AgentRunnerConfigError` at construction.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mollog import get_logger

from molexp.agent.compaction import CompactionSettings
from molexp.agent.events import (
    AgentEvent,
    AsyncIteratorEventSink,
    LoopCompletedEvent,
    LoopStartedEvent,
)
from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.loop import AgentRunResult
from molexp.agent.loops._compact import maybe_compact
from molexp.agent.router import ModelTier, Router, TierModels
from molexp.agent.runtime import AgentRuntime
from molexp.agent.session import Session
from molexp.agent.session_storage import (
    InMemorySessionStorage,
    JsonlSessionStorage,
)
from molexp.agent.types import Message

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool

    from molexp.agent.folders import Agent as AgentFolder


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
    """Drive one text call or one ReAct per :meth:`run`.

    Construction performs no network IO — the underlying pydantic-ai
    ``Agent``\\ s are built lazily on first :meth:`run`.
    """

    def __init__(
        self,
        *,
        model: str | object | None = None,
        models: Mapping[ModelTier | str, str | object] | None = None,
        router: Router | None = None,
        tools: tuple[Tool[None] | Callable[..., Any], ...] = (),
        workspace: Path | None = None,
        session_anchor: Path | None = None,
        system_prompt: str = "",
        context_block: str = "",
        operation_mode: str = "chat",
        mode: Literal["text", "agentic"] = "agentic",
        name: str = "agent",
        compaction: CompactionSettings | None = None,
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

        self.tools = tools
        self.workspace = workspace
        self.session_anchor = session_anchor
        self.system_prompt = system_prompt
        self.context_block = context_block
        self.operation_mode = operation_mode
        self.mode = mode
        self.name = name
        self.compaction = compaction or CompactionSettings()
        self._router: Router | None = router
        self._tier_models: TierModels | None
        if router is not None:
            self._tier_models = None
        elif model is not None:
            self._tier_models = dict.fromkeys(ModelTier, model)
        else:
            assert models is not None
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

    async def run(self, session: Session, user_input: str) -> AgentRunResult:
        """Drive one call and return its terminal :class:`AgentRunResult`."""
        accumulated: list[AgentEvent] = []
        async for event in self.run_events(session, user_input):
            accumulated.append(event)
        return _result_from_stream(tuple(accumulated))

    async def run_events(self, session: Session, user_input: str) -> AsyncIterator[AgentEvent]:
        """Drive one call and yield its :data:`AgentEvent` stream live."""
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
                if self.mode == "text":
                    await self._run_text(runtime=runtime, sink=sink, user_input=user_input)
                else:
                    from molexp.agent.react import run_react_turn

                    workspace = self.workspace if self.workspace is not None else Path.cwd()
                    await run_react_turn(
                        runtime=runtime,
                        sink=sink,
                        user_input=user_input,
                        workspace=workspace,
                        operation_mode=self.operation_mode,
                        system_prompt=self.system_prompt,
                        context_block=self.context_block,
                        extra_tools=tuple(self.tools),
                        compaction=self.compaction,
                        name=self.name,
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

    async def _run_text(
        self,
        *,
        runtime: AgentRuntime,
        sink: AsyncIteratorEventSink,
        user_input: str,
    ) -> None:
        """One ``complete_text`` — Chat must not loop."""
        await sink(LoopStartedEvent(loop_name=self.name, user_input=user_input))
        runtime.router.clear_usage()
        runtime.session.append_message(Message(role="user", content=user_input))
        await maybe_compact(
            runtime=runtime,
            sink=sink,
            settings=self.compaction,
            loop_name=self.name,
        )
        from molexp.agent.react import load_model_history

        result = await runtime.router.complete_text(
            prompt=user_input,
            system=self.system_prompt,
            message_history=load_model_history(runtime.session),
        )
        runtime.session.append_message(Message(role="assistant", content=result.text))
        breakdown = runtime.router.snapshot_usage()
        run_result = AgentRunResult(
            text=result.text,
            messages=runtime.session.build_context(),
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        await sink(
            LoopCompletedEvent(
                text=result.text,
                result=run_result.model_dump(mode="json"),
            )
        )

    def session(self, session_id: str) -> Session:
        """Return a :class:`Session` named ``session_id``."""
        directory = self._session_directory(session_id)
        if directory is not None:
            return Session(storage=JsonlSessionStorage(directory), session_id=session_id)
        return Session(storage=InMemorySessionStorage(), session_id=session_id)

    def _session_directory(self, session_id: str) -> Path | None:
        """Return the on-disk anchor dir for ``session_id``, or ``None``."""
        agent_folder = self._ensure_agent_folder()
        if agent_folder is None:
            return None
        try:
            if agent_folder.has_session(session_id):
                sess_folder = agent_folder.get_session(session_id)
            else:
                sess_folder = agent_folder.add_session(session_id)
            return Path(str(sess_folder.path()))
        except OSError as exc:  # pragma: no cover
            _LOG.warning(
                f"[runner] session({session_id!r}): could not open on-disk "
                f"anchor ({exc!r}); using in-memory storage."
            )
            return None

    def _ensure_agent_folder(self) -> AgentFolder | None:
        """Lazily mount the persistent :class:`Agent` folder for this runner."""
        if self._agent_folder is not None:
            return self._agent_folder
        anchor = self.session_anchor if self.session_anchor is not None else self.workspace
        if anchor is None:
            return None
        try:
            from molexp.agent.folders import Agent as AgentFolder

            self._agent_folder = AgentFolder(name=self.name, root_path=Path(anchor))
        except OSError as exc:
            _LOG.warning(
                f"[runner] could not open Agent folder for {anchor!r}: "
                f"{exc!r}; sessions will be in-memory only."
            )
            return None
        return self._agent_folder

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
        """Construct the :class:`LocalExecutionEnv` for this runner."""
        if self.workspace is not None:
            scratch = Path(self.workspace) / self.name / ".scratch"
        else:
            import tempfile

            scratch = Path(tempfile.gettempdir()) / "molexp-agent-scratch"
        return LocalExecutionEnv(scratch_dir=scratch)

    def _compose_system_prompt(self) -> str:
        """Concatenate MCP ``usage_instructions`` + the workspace path note."""
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


def _result_from_stream(events: tuple[AgentEvent, ...]) -> AgentRunResult:
    """Fold an accumulated event stream into the terminal :class:`AgentRunResult`."""
    terminal: LoopCompletedEvent | None = None
    for event in events:
        if isinstance(event, LoopCompletedEvent):
            terminal = event
    if terminal is None:
        raise RuntimeError(
            "the event stream ended without a LoopCompletedEvent; "
            "every AgentRunner turn must emit one as its terminal event."
        )
    if terminal.result is not None:
        payload = dict(terminal.result)
        payload.pop("events", None)
        base = AgentRunResult.model_validate(payload)
    else:
        base = AgentRunResult(text=terminal.text)
    return base.model_copy(update={"events": events})


def _normalize_tier_map(
    raw: Mapping[ModelTier | str, str | object],
) -> dict[ModelTier, str | object]:
    """Coerce string keys (``"cheap"``) to :class:`ModelTier` and validate coverage."""
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
