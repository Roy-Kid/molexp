"""Tests for the ``Stage`` ABC introduced by agent-mode-stage-pipeline-01.

Covers ac-001 of the substrate spec: ``Stage`` is a plain Generic ABC
(NOT pydantic), with an abstract async-generator ``run`` method and
the four declared ``ClassVar`` attributes (``name`` / ``pre_state`` /
``post_state`` / ``persists``).
"""

from __future__ import annotations

import inspect
from typing import ClassVar

import pytest
from pydantic import BaseModel

from molexp.agent.harness.stage import Stage


def test_stage_is_not_a_pydantic_basemodel() -> None:
    """Stage must NOT be a pydantic BaseModel — it holds runtime instances."""
    assert not issubclass(Stage, BaseModel)


def test_stage_is_abstract_with_async_run() -> None:
    """Stage.run is decorated @abstractmethod; instantiation is forbidden."""
    assert getattr(Stage.run, "__isabstractmethod__", False) is True
    with pytest.raises(TypeError):
        Stage()  # type: ignore[abstract]


def test_stage_classvar_defaults_declared() -> None:
    """``pre_state`` / ``post_state`` default to None; ``persists`` defaults to ()."""
    assert Stage.pre_state is None
    assert Stage.post_state is None
    assert Stage.persists == ()


def test_stage_classvar_annotations_carry_classvar_marker() -> None:
    """``name`` / ``pre_state`` / ``post_state`` / ``persists`` are ClassVars."""
    raw_hints = inspect.get_annotations(Stage)
    # ClassVar annotations survive on the raw class annotations as strings
    # under `from __future__ import annotations` — assert by substring.
    for field in ("name", "pre_state", "post_state", "persists"):
        annotation = raw_hints.get(field, "")
        assert "ClassVar" in str(annotation), (
            f"Stage.{field} annotation '{annotation}' must be a ClassVar"
        )


def test_concrete_subclass_runnable() -> None:
    """A concrete subclass that implements ``run`` can be instantiated."""

    class _ConcreteStage(Stage[str, str]):
        name: ClassVar[str] = "concrete"

        async def run(self, *, harness, input):
            yield input

    instance = _ConcreteStage()
    assert instance.name == "concrete"


def test_subclass_can_override_classvar_defaults() -> None:
    """Subclasses set their own ``pre_state`` / ``post_state`` / ``persists``."""

    class _StatefulStage(Stage[None, None]):
        name: ClassVar[str] = "stateful"
        pre_state: ClassVar[str | None] = "intake"
        post_state: ClassVar[str | None] = "exploring"
        persists: ClassVar[tuple[str, ...]] = ("intent.json",)

        async def run(self, *, harness, input):
            yield None

    assert _StatefulStage.pre_state == "intake"
    assert _StatefulStage.post_state == "exploring"
    assert _StatefulStage.persists == ("intent.json",)


def test_run_signature_requires_keyword_only_harness_and_input() -> None:
    """``Stage.run(self, *, harness, input)`` — both kwargs-only."""
    sig = inspect.signature(Stage.run)
    params = sig.parameters
    assert "harness" in params
    assert "input" in params
    assert params["harness"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["input"].kind is inspect.Parameter.KEYWORD_ONLY
