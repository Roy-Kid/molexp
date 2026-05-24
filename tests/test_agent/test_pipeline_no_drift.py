"""No-drift guard for declarative mode pipelines (mode-pipeline-flowchart ac-008).

Each mode declares its stage topology as a ``ModePipeline`` class
attribute. This mechanical test keeps that declaration honest: it
AST-extracts every ``harness.stage("...")`` string literal from the
mode's run-source and asserts the set equals ``Mode.pipeline.stages``.

The load-bearing nuance: AuthorMode's ``harness.stage(...)`` calls live
in ``author/materialize.py``, not ``author/_mode.py`` — so its case is
parametrized against ``materialize.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import molexp.agent.modes.author.materialize as _author_materialize
import molexp.agent.modes.chat as _chat
import molexp.agent.modes.interactive.mode as _interactive_mode
import molexp.agent.modes.plan._mode as _plan_mode
import molexp.agent.modes.review._mode as _review_mode
import molexp.agent.modes.run._mode as _run_mode
from molexp.agent.mode import AgentMode
from molexp.agent.modes import (
    AuthorMode,
    ChatMode,
    InteractiveMode,
    PlanMode,
    ReviewMode,
    RunMode,
)


def _harness_stage_literals(module_path: Path) -> set[str]:
    """Return every string literal passed to a ``harness.stage("...")`` call."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    literals: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "stage"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "harness"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            literals.add(node.args[0].value)
    return literals


_CASES = [
    pytest.param(ChatMode, _chat.__file__, id="chat"),
    pytest.param(PlanMode, _plan_mode.__file__, id="plan"),
    # AuthorMode's harness.stage(...) calls live in materialize.py.
    pytest.param(AuthorMode, _author_materialize.__file__, id="author"),
    pytest.param(RunMode, _run_mode.__file__, id="run"),
    pytest.param(ReviewMode, _review_mode.__file__, id="review"),
    pytest.param(InteractiveMode, _interactive_mode.__file__, id="interactive"),
]


@pytest.mark.parametrize(("mode_cls", "source_path"), _CASES)
def test_pipeline_stages_have_no_drift(mode_cls: type[AgentMode], source_path: str) -> None:
    # After agent-mode-stage-pipeline-01, ``pipeline.stages`` is a tuple
    # of ``Stage`` instances; extract their declared names for comparison.
    declared = {stage.name for stage in mode_cls.pipeline.stages}
    actual = _harness_stage_literals(Path(source_path))
    assert declared == actual, (
        f"{mode_cls.__name__}: declared pipeline.stages {sorted(declared)} "
        f"!= harness.stage() literals {sorted(actual)} in {source_path}"
    )
