"""No-drift guard for declarative mode pipelines.

Each mode declares its stage topology as a :class:`ModePipeline` class
attribute. This mechanical test keeps that declaration honest:

- **Pre-migration modes** (chat / author / run / review / interactive
  for now): the substrate's executor still uses
  ``async with harness.stage("name"):`` brackets inside the mode's
  ``run`` body. The test AST-extracts every literal and asserts the
  set matches ``Mode.pipeline.stages``.
- **Post-migration modes** (PlanMode after
  ``agent-mode-stage-pipeline-02``): all ``harness.stage(...)``
  brackets are emitted by
  :func:`~molexp.agent.pipeline.execute_pipeline`, reading
  each :class:`Stage.name` ClassVar dynamically. The literals move
  from the mode source into per-stage subclass ``name: ClassVar[str] =
  "..."`` declarations. The test scans the entire
  ``modes/<verb>/stages/`` package for those declarations and asserts
  the same equality.

The load-bearing nuance: AuthorMode's pre-migration
``harness.stage(...)`` calls live in ``author/materialize.py``, not
``author/_mode.py`` — so its case is parametrized against
``materialize.py``.
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


def _harness_stage_literals(source_paths: tuple[Path, ...]) -> set[str]:
    """Return every literal passed to a ``harness.stage("...")`` call.

    Recurses over each path that is a directory, parsing every ``.py``
    file under it.
    """
    literals: set[str] = set()
    files: list[Path] = []
    for path in source_paths:
        if path.is_dir():
            files.extend(path.rglob("*.py"))
        else:
            files.append(path)
    for fp in files:
        tree = ast.parse(fp.read_text(encoding="utf-8"))
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


def _stage_name_classvars(source_paths: tuple[Path, ...]) -> set[str]:
    """Return every ``name: ClassVar[str] = "..."`` declaration's literal.

    Walks the AST for any ``AnnAssign`` whose target is named ``name``
    and whose value is a string constant. Used to catch the names
    declared by ``Stage`` subclasses under a mode's ``stages/`` package
    after the substrate migration moves the literals there.
    """
    literals: set[str] = set()
    files: list[Path] = []
    for path in source_paths:
        if path.is_dir():
            files.extend(path.rglob("*.py"))
        else:
            files.append(path)
    for fp in files:
        tree = ast.parse(fp.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "name"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                literals.add(node.value.value)
    return literals


# Each case's ``source_paths`` is a tuple of files / directories the
# helper scans for either ``harness.stage("...")`` literals (legacy
# pattern) or ``name: ClassVar[str] = "..."`` declarations (substrate
# Stage subclasses). Post-substrate migration each mode's source moves
# from ``_mode.py``'s ``async with harness.stage(...):`` brackets into
# per-mode Stage subclasses under a sibling ``stages.py`` / ``stages/``
# package.
_plan_stages_dir = Path(_plan_mode.__file__).parent / "stages"
_chat_stages_file = Path(_chat.__file__).parent / "chat_stages.py"
_run_stages_file = Path(_run_mode.__file__).parent / "stages.py"
_review_stages_file = Path(_review_mode.__file__).parent / "stages.py"
_interactive_stages_file = Path(_interactive_mode.__file__).parent / "stages.py"
_author_stages_file = Path(_author_materialize.__file__).parent / "stages.py"

_CASES = [
    pytest.param(ChatMode, (Path(_chat.__file__), _chat_stages_file), id="chat"),
    pytest.param(
        PlanMode,
        (Path(_plan_mode.__file__), _plan_stages_dir),
        id="plan",
    ),
    # AuthorMode's per-stage bodies live in ``stages.py`` post-migration;
    # ``materialize.py`` still carries the pre-migration literal references.
    pytest.param(
        AuthorMode,
        (Path(_author_materialize.__file__), _author_stages_file),
        id="author",
    ),
    pytest.param(RunMode, (Path(_run_mode.__file__), _run_stages_file), id="run"),
    pytest.param(
        ReviewMode,
        (Path(_review_mode.__file__), _review_stages_file),
        id="review",
    ),
    pytest.param(
        InteractiveMode,
        (Path(_interactive_mode.__file__), _interactive_stages_file),
        id="interactive",
    ),
]


@pytest.mark.parametrize(("mode_cls", "source_paths"), _CASES)
def test_pipeline_stages_have_no_drift(
    mode_cls: type[AgentMode], source_paths: tuple[Path, ...]
) -> None:
    declared = {stage.name for stage in mode_cls.pipeline.stages}
    literal_stage_calls = _harness_stage_literals(source_paths)
    classvar_stage_names = _stage_name_classvars(source_paths)
    actual = literal_stage_calls | classvar_stage_names
    assert declared == actual, (
        f"{mode_cls.__name__}: declared pipeline.stages {sorted(declared)} "
        f"!= harness.stage() literals + Stage.name ClassVars {sorted(actual)} "
        f"in {source_paths}"
    )
