"""AgentRunner / AgentStateStore plain-class construction tests.

Both types carry live runtime references (model client, registry,
session store, recovery policy, callable evaluators, etc.) and are
plain Python classes — not BaseModel, not @dataclass, no
``arbitrary_types_allowed=True``.
"""

from __future__ import annotations

import dataclasses

from molexp.agent.memory import NoopMemoryStore
from molexp.agent.orchestration.runner import AgentRunner
from molexp.agent.service import AgentStateStore
from molexp.agent.sessions import SessionStore
from molexp.agent.skills import SkillStore
from molexp.agent.testing import FakeModelClient
from molexp.agent.tools.dispatcher import ToolDispatcher
from molexp.agent.tools.registry import ToolRegistry


class TestAgentRunnerConstruction:
    def test_required_only_kwargs(self, tmp_path):
        runner = AgentRunner(
            model=FakeModelClient(),
            registry=ToolRegistry(),
            store=SessionStore(tmp_path / "sessions"),
        )
        assert runner.model is not None
        assert runner.registry is not None
        assert runner.store is not None
        # lazy dispatcher default
        assert runner.dispatcher is not None
        assert isinstance(runner.dispatcher, ToolDispatcher)

    def test_explicit_dispatcher_kwarg(self, tmp_path):
        registry = ToolRegistry()
        custom = ToolDispatcher(registry)
        runner = AgentRunner(
            model=FakeModelClient(),
            registry=registry,
            store=SessionStore(tmp_path / "sessions"),
            dispatcher=custom,
        )
        assert runner.dispatcher is custom

    def test_runner_is_not_dataclass_not_basemodel(self):
        # Plain class — neither stdlib dataclass nor pydantic BaseModel
        assert not dataclasses.is_dataclass(AgentRunner)
        from pydantic import BaseModel

        assert not issubclass(AgentRunner, BaseModel)


class TestAgentStateStoreConstruction:
    def test_explicit_init(self, tmp_path):
        sessions = SessionStore(tmp_path / "sessions")
        skills = SkillStore(tmp_path)
        memory = NoopMemoryStore()
        state = AgentStateStore(sessions=sessions, skills=skills, memory=memory)
        assert state.sessions is sessions
        assert state.skills is skills
        assert state.memory is memory

    def test_state_store_is_not_dataclass_not_basemodel(self):
        assert not dataclasses.is_dataclass(AgentStateStore)
        from pydantic import BaseModel

        assert not issubclass(AgentStateStore, BaseModel)
