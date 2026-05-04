"""End-to-end plan-mode handoff: exit_plan_mode → PlanCreatedEvent → approve → continue.

Plan-mode in molexp emits ONE kind of plan: a runnable workflow. Every
numbered step in ``plan_markdown`` corresponds to a node in
``workflow_preview.workflow_ir.task_configs``. On approval the session
flips out of plan mode and rebuilds the agent for the post-plan
toolset.

These tests bypass the LLM by driving the session directly — the same
code path ``exit_plan_mode`` exercises — to keep them deterministic
and offline.
"""

from __future__ import annotations

import asyncio

import pytest

from molexp.plugins.agent_pydanticai._pydantic_ai.session import PydanticAISession
from molexp.plugins.agent_pydanticai.types import Goal, PlanCreatedEvent


def _valid_preview() -> dict:
    return {
        "mermaid": "flowchart LR\n  A --> B",
        "workflow_ir": {
            "name": "demo",
            "task_configs": [
                {"task_id": "t1", "task_type": "noop", "config": {}},
            ],
            "links": [],
        },
        "intervention_points": ["rename A"],
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plan_approval_disables_plan_mode_and_rebuilds(workspace):
    goal = Goal(description="explore", plan_mode=True)
    session = PydanticAISession(
        session_id="sess-test", goal=goal, workspace=workspace
    )

    rebuild_calls: list[Goal] = []

    def _rebuild(updated_goal: Goal):
        rebuild_calls.append(updated_goal)
        return object()  # stand-in for a fresh Agent

    session._rebuild_agent = _rebuild

    decision_task = asyncio.create_task(
        session.await_plan_decision(
            plan_markdown="1. inspect\n2. plan\n",
            workflow_preview=_valid_preview(),
        )
    )
    # Yield once so the event lands on the queue.
    await asyncio.sleep(0)

    pending = session.drain_pending_events()
    plan_events = [e for e in pending if isinstance(e, PlanCreatedEvent)]
    assert len(plan_events) == 1
    plan_event = plan_events[0]
    assert plan_event.plan_markdown.startswith("1. inspect")
    assert plan_event.workflow_preview is not None
    assert plan_event.workflow_preview.mermaid.startswith("flowchart")
    assert plan_event.workflow_preview.workflow_ir["name"] == "demo"
    assert plan_event.workflow_preview.intervention_points == ["rename A"]
    # Auto-rendered Python script always present, derivable from the IR.
    script = plan_event.workflow_preview.python_script
    assert "WORKFLOW_IR" in script
    # repr() uses single quotes; compare on the unquoted token.
    assert "task_id" in script
    assert "t1" in script

    # Approve with an edit.
    await session.respond_plan(
        request_id=plan_event.request_id,
        approved=True,
        edited_plan="1. inspect (edited)\n2. plan\n",
        edited_workflow_ir={"name": "demo-v2"},
    )

    decision = await asyncio.wait_for(decision_task, timeout=1.0)
    assert decision == {
        "approved": True,
        "edited_plan": "1. inspect (edited)\n2. plan\n",
        "edited_workflow_ir": {"name": "demo-v2"},
    }
    assert goal.plan_mode is False
    assert len(rebuild_calls) == 1
    assert rebuild_calls[0].plan_mode is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plan_without_preview_is_rejected(workspace):
    goal = Goal(description="explore", plan_mode=True)
    session = PydanticAISession(
        session_id="sess-test", goal=goal, workspace=workspace
    )

    with pytest.raises(ValueError, match="workflow_preview"):
        await session.await_plan_decision(
            plan_markdown="1. run",
            workflow_preview=None,  # type: ignore[arg-type]
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plan_with_empty_task_configs_is_rejected(workspace):
    """Empty IR must not park the session.

    Investigation-style plans must encode their steps as nodes, not as
    a degenerate empty workflow.
    """
    goal = Goal(description="explore", plan_mode=True)
    session = PydanticAISession(
        session_id="sess-test", goal=goal, workspace=workspace
    )

    with pytest.raises(ValueError, match="task_configs"):
        await session.await_plan_decision(
            plan_markdown="1. read",
            workflow_preview={
                "workflow_ir": {"name": "x", "task_configs": [], "links": []},
            },
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plan_decision_rejection_keeps_plan_mode(workspace):
    goal = Goal(description="explore", plan_mode=True)
    session = PydanticAISession(
        session_id="sess-test", goal=goal, workspace=workspace
    )
    session._rebuild_agent = lambda g: None

    decision_task = asyncio.create_task(
        session.await_plan_decision(
            plan_markdown="1. plan",
            workflow_preview=_valid_preview(),
        )
    )
    await asyncio.sleep(0)

    plan_event = next(
        e for e in session.drain_pending_events() if isinstance(e, PlanCreatedEvent)
    )

    await session.respond_plan(
        request_id=plan_event.request_id,
        approved=False,
        feedback="please scope this to project X",
    )
    decision = await asyncio.wait_for(decision_task, timeout=1.0)
    assert decision["approved"] is False
    assert "scope this" in decision["feedback"]
    # Goal stays in plan mode — agent revises and tries again.
    assert goal.plan_mode is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plan_decision_unknown_request_id_is_no_op(workspace):
    """Resolving a stale request_id should be a no-op, not a crash."""
    goal = Goal(description="explore", plan_mode=True)
    session = PydanticAISession(
        session_id="sess-test", goal=goal, workspace=workspace
    )
    await session.respond_plan(
        request_id="nonexistent",
        approved=True,
    )
    # Goal stayed unchanged because no future was actually resolved.
    assert goal.plan_mode is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plan_event_carries_python_script_when_agent_supplies_one(workspace):
    """Agent-supplied python_script is preserved verbatim, not regenerated."""
    goal = Goal(description="explore", plan_mode=True)
    session = PydanticAISession(
        session_id="sess-test", goal=goal, workspace=workspace
    )
    session._rebuild_agent = lambda g: None

    custom_script = "# hand-authored\nfrom molexp.workflow.spec import WorkflowSpec\n"
    decision_task = asyncio.create_task(
        session.await_plan_decision(
            plan_markdown="1. plan",
            workflow_preview={
                **_valid_preview(),
                "python_script": custom_script,
            },
        )
    )
    await asyncio.sleep(0)

    plan_event = next(
        e for e in session.drain_pending_events() if isinstance(e, PlanCreatedEvent)
    )
    assert plan_event.workflow_preview is not None
    assert plan_event.workflow_preview.python_script == custom_script

    await session.respond_plan(
        request_id=plan_event.request_id, approved=True
    )
    await asyncio.wait_for(decision_task, timeout=1.0)
