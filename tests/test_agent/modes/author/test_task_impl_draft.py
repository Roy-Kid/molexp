"""Schema-level invariants for :class:`TaskImplDraft` + ``assemble_impl_module``.

The draft schema is body-only:

- ``imports`` — each entry is exactly one ``import`` / ``from … import …``
  statement. Banning embedded ``\\n`` prevents arbitrary multi-statement
  injection into the assembled module.
- ``body`` — the Python statements that compose the imports to bind the
  local names declared in ``PlanStep.io.outputs``.
- ``is_stub`` — when ``True`` the assembler emits
  ``raise NotImplementedError`` and drops the imports / return.

Everything else (function name, docstring, input bindings, return shape)
is derived from the :class:`PlanStep` by ``assemble_impl_module(draft, step)``.
"""

from __future__ import annotations

import ast

import pytest
from pydantic import ValidationError

from molexp.agent.modes.author.codegen import (
    TaskImplDraft,
    assemble_impl_module,
)

from .conftest import make_step

# ── imports field validator ──────────────────────────────────────────────


def test_imports_rejects_embedded_newline_statement() -> None:
    """The bug the validator exists for: a single entry containing
    ``\\n`` would splice arbitrary statements past every gate."""
    with pytest.raises(ValidationError):
        TaskImplDraft(imports=("import os\nos.system('rm -rf /')",))


def test_imports_rejects_non_import_statement() -> None:
    with pytest.raises(ValidationError):
        TaskImplDraft(imports=("x = 1",))


def test_imports_rejects_unparseable_entry() -> None:
    with pytest.raises(ValidationError):
        TaskImplDraft(imports=("import !!! foo",))


@pytest.mark.parametrize(
    "line",
    [
        "import os",
        "from molpy import System",
        "from molpy.typifier import OplsTypifier",
        "import numpy as np",
    ],
)
def test_imports_accepts_single_import_statement(line: str) -> None:
    TaskImplDraft(imports=(line,))


# ── assemble_impl_module: structural shape ───────────────────────────────


def test_assemble_emits_async_function_named_after_step() -> None:
    """Function name = ``module_id(step.id)`` — derived, not LLM-supplied."""
    step = make_step("step-1-build-chain")
    draft = TaskImplDraft(body="return")
    source = assemble_impl_module(draft, step)
    tree = ast.parse(source)
    funcs = [s for s in tree.body if isinstance(s, ast.AsyncFunctionDef)]
    assert len(funcs) == 1
    assert funcs[0].name == "step_1_build_chain"
    # The function takes ctx and nothing else.
    assert [a.arg for a in funcs[0].args.args] == ["ctx"]


def test_assemble_uses_composition_notes_as_docstring() -> None:
    """Docstring is derived from PlanStep, not from the draft."""
    step = make_step("foo", composition_notes="Build a polymer using molpy.")
    draft = TaskImplDraft(body="return")
    source = assemble_impl_module(draft, step)
    tree = ast.parse(source)
    module_doc = ast.get_docstring(tree)
    assert module_doc == "Build a polymer using molpy."


def test_assemble_auto_binds_inputs_from_planstep_single_dep() -> None:
    """With one upstream, ``ctx.inputs`` is the upstream return dict."""
    from molexp.agent.modes._planning import PlanStepInput

    step = make_step(
        "downstream",
        depends_on=("upstream",),
        inputs=(PlanStepInput(name="peo_chain", source_step="upstream"),),
    )
    draft = TaskImplDraft(body="return")
    source = assemble_impl_module(draft, step)
    assert "peo_chain = ctx.inputs['peo_chain']" in source


def test_assemble_auto_binds_inputs_from_planstep_multi_dep() -> None:
    """With multiple upstreams, ``ctx.inputs`` is keyed by step id first."""
    from molexp.agent.modes._planning import PlanStepInput

    step = make_step(
        "downstream",
        depends_on=("up1", "up2"),
        inputs=(
            PlanStepInput(name="forcefield", source_step="up1"),
            PlanStepInput(name="chain", source_step="up2"),
        ),
    )
    draft = TaskImplDraft(body="return")
    source = assemble_impl_module(draft, step)
    assert "forcefield = ctx.inputs['up1']['forcefield']" in source
    assert "chain = ctx.inputs['up2']['chain']" in source


def test_assemble_auto_generates_return_from_planstep_outputs() -> None:
    """Return shape is derived from PlanStep.io.outputs — model never writes it."""
    step = make_step("build-chain", outputs=("peo_chain",))
    draft = TaskImplDraft(
        imports=("from molpy.tool.polymer import polymer",),
        body='peo_chain = polymer("{[<]CCO[>]}|3|")',
    )
    source = assemble_impl_module(draft, step)
    assert "return {'peo_chain': peo_chain}" in source


def test_assemble_emits_no_imports_when_stub() -> None:
    """A stub never reaches its imports; dropping them avoids a hard
    ImportError when the api_ref isn't available in the host env."""
    step = make_step("foo")
    draft = TaskImplDraft(
        imports=("from molpy.optional_extension import Foo",),
        is_stub=True,
    )
    source = assemble_impl_module(draft, step)
    assert "molpy.optional_extension" not in source
    assert "raise NotImplementedError" in source


def test_assemble_stub_body_uses_step_id() -> None:
    step = make_step("step-1-load-ff")
    draft = TaskImplDraft(is_stub=True)
    source = assemble_impl_module(draft, step)
    ast.parse(source)  # always parseable
    assert "step-1-load-ff is a stub" in source


def test_assemble_handles_quotes_in_step_id() -> None:
    """Step id with awkward chars — ``repr()`` makes the literal safe."""
    step = make_step('odd"step')
    draft = TaskImplDraft(is_stub=True)
    source = assemble_impl_module(draft, step)
    ast.parse(source)


def test_assemble_sanitises_non_identifier_output_name() -> None:
    """A free-form output name like ``data.peo`` becomes a Python-safe
    local (``data_peo``) while keeping the original as the dict key
    so downstream consumers still look it up under the planner's name."""
    step = make_step("write", outputs=("data.peo",))
    draft = TaskImplDraft(body="data_peo = 'data.peo'")
    source = assemble_impl_module(draft, step)
    ast.parse(source)
    assert "return {'data.peo': data_peo}" in source


def test_assemble_sanitises_non_identifier_input_name() -> None:
    """Same idea for inputs — the body sees a Python-safe local."""
    from molexp.agent.modes._planning import PlanStepInput

    step = make_step(
        "consume",
        depends_on=("upstream",),
        inputs=(PlanStepInput(name="data.peo", source_step="upstream"),),
    )
    draft = TaskImplDraft(body="x = data_peo")
    source = assemble_impl_module(draft, step)
    ast.parse(source)
    assert "data_peo = ctx.inputs['data.peo']" in source


def test_assemble_handles_triple_quote_in_composition_notes() -> None:
    """Docstring containing ``\\\"\\\"\\\"`` is escaped so the literal terminates."""
    step = make_step("foo", composition_notes='See """example""" for usage.')
    draft = TaskImplDraft(body="return")
    source = assemble_impl_module(draft, step)
    ast.parse(source)
