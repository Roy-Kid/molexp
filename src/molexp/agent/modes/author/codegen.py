"""AuthorMode codegen stages — recreated against the typed plan surface.

The codegen nodes the old ``plan/tasks.py`` owned (``CompileWorkflowIR``,
``CompileTaskIR``, ``GenerateWorkflowSkeleton``, ``GenerateTaskTests``,
``GenerateTaskImplementations``, ``ValidateWorkspace``,
``FinalHandoffCheck``) are recreated here as **plain async functions** —
AuthorMode's own pipeline is a plain async stage sequence on the harness,
not a ``pydantic_graph`` graph. Each function reads the typed
:class:`~molexp.agent.modes._planning.PlanGraph` plus the lowered
:class:`~molexp.workflow.WorkflowContract`; the un-evidenced-symbol gate
(:func:`~molexp.agent.modes.author.codegen_evidence.validate_codegen_evidence`)
keys off the union of every :class:`PlanStep`'s inline ``api_refs``.

Capability evidence lives on each :class:`PlanStep` (``api_refs`` +
``composition_notes``); there is no separate graph artefact.

The LLM-bearing functions (``compile_task_ir``, ``generate_task_tests``,
``generate_task_implementations``) go through the
:class:`~molexp.agent.router.Router` structured path; the structural
renderers (``generate_workflow_skeleton``) are templated.
"""

from __future__ import annotations

import ast
import asyncio

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from molexp.agent.modes._planning import PlanGraph, PlanStep
from molexp.agent.modes.author.codegen_evidence import (
    MissingCapability,
    validate_codegen_evidence,
)
from molexp.agent.modes.author.renderers import (
    PACKAGE_INIT_BODY,
    TASKS_INIT_BODY,
    module_id,
    render_stub_test,
    render_workflow_module,
    render_workflow_structure_test,
    validate_python,
)
from molexp.agent.modes.author.workspace_layout import MaterializedLayout
from molexp.agent.router import ModelTier, Router
from molexp.workflow import (
    TaskIO,
    WorkflowContract,
    default_compiler,
    validate_workflow_contract,
)

__all__ = [
    "CodegenError",
    "GeneratedModule",
    "RepairDecision",
    "TaskIRBrief",
    "TaskImplDraft",
    "WorkspaceValidation",
    "assemble_impl_module",
    "compile_task_ir",
    "generate_task_implementations",
    "generate_task_tests",
    "generate_workflow_skeleton",
    "validate_assembled_impl",
    "validate_test_source",
    "validate_workspace",
    "write_workflow_ir",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")

_GATE_RETRY_BUDGET = 3
"""Total attempts in :func:`_generate_one` before surfacing a CodegenError.
Each failed attempt feeds the gate verdict back into the next prompt so
the model fixes the specific issue (missing import, hallucinated symbol,
test code in impl path, …) on retry. Three attempts has been enough in
practice to recover from one-off slips without inviting a runaway loop."""


class CodegenError(RuntimeError):
    """Raised when a codegen stage produces output that fails its gate.

    Carries the structured failure cause: a syntax error in generated
    source, or a tuple of un-evidenced :class:`MissingCapability` rows.
    """

    def __init__(self, message: str, *, missing: tuple[MissingCapability, ...] = ()) -> None:
        super().__init__(message)
        self.missing = missing


# ── Structured codegen schemas ───────────────────────────────────────────


class TaskIRBrief(BaseModel):
    """The per-task IR brief the LLM drafts for one workflow task.

    Attributes:
        task_id: The task's stable id (must match the contract entry).
        responsibility: One sentence on what the task does.
        success_criteria: What a successful run produces.
        test_expectations: What the generated test should assert.
        is_stub: True when no capability evidence supports the task.
    """

    model_config = _FROZEN

    task_id: str
    responsibility: str
    success_criteria: tuple[str, ...] = ()
    test_expectations: tuple[str, ...] = ()
    is_stub: bool = False


class GeneratedModule(BaseModel):
    """One LLM-generated Python module (implementation or test).

    Attributes:
        task_id: The task this module belongs to.
        source: The full module source text.
        is_stub: True when the module is a no-op stub.
    """

    model_config = _FROZEN

    task_id: str
    source: str
    is_stub: bool = False


class TaskImplDraft(BaseModel):
    """Body-only draft for one workflow task.

    The codegen layer derives EVERYTHING structural — the function
    name (from ``PlanStep.id``), the docstring (from
    ``PlanStep.composition_notes``), the input bindings (from
    ``PlanStep.io.inputs``), and the return shape (from
    ``PlanStep.io.outputs``). The model contributes only what
    varies task-to-task: which symbols to import and the handful
    of statements that compose them. The assembled module is:

    .. code-block:: python

        \"\"\"<from PlanStep.composition_notes>\"\"\"

        <imports>


        async def <module_id(step.id)>(ctx):
            <input_name1> = ctx.inputs[...]   # auto from PlanStep.io.inputs
            <input_name2> = ctx.inputs[...]   # auto

            <body — the model writes this>

            return {"<output1>": <output1>, ...}   # auto from PlanStep.io.outputs

    The shape gate enforces that ONLY this structure can be emitted —
    no module-level mutation, no class declarations, no test code —
    so anti-patterns like monkey-patching upstream classes have no
    surface to land in.

    Attributes:
        imports: Import lines the body needs. Each entry MUST be exactly
            one ``import X`` or ``from X import Y`` statement.
            Re-exports of an ``api_ref`` at a shorter path are fine;
            anything else is rejected by the codegen-evidence gate.
        body: The Python statements that compose the imports to bind the
            local names matching ``PlanStep.io.outputs``. NO leading
            indentation; the assembler indents under the function.
        is_stub: True when the matching PlanStep has no api_refs to
            ground a real impl; the assembler emits ``raise
            NotImplementedError``.
    """

    model_config = _FROZEN

    imports: tuple[str, ...] = ()
    body: str = ""
    is_stub: bool = False

    @field_validator("imports")
    @classmethod
    def _check_imports(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Each entry must parse to exactly one ``import`` / ``from … import …``.

        Without this check an entry containing ``\\n`` could splice
        arbitrary statements into the assembled module: one-statement-
        per-entry is the schema contract.
        """
        for line in value:
            try:
                tree = ast.parse(line)
            except SyntaxError as exc:
                raise ValueError(
                    f"imports entry is not parseable Python: {line!r} — {exc.msg}"
                ) from exc
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Import | ast.ImportFrom):
                raise ValueError(
                    "each imports entry must be exactly one `import` or "
                    f"`from … import …` statement; got {line!r}"
                )
        return value


class RepairDecision(BaseModel):
    """A targeted debug-loop repair that may rewrite the impl, the test,
    or both.

    The debug loop runs a generated impl against its generated test in
    an isolated subprocess. When the test fails, the failure may live
    on either side: the impl might call a non-existent API (impl bug),
    OR the test might construct a broken synthetic input / over-
    specify the contract (test bug). The repair agent inspects the
    failure and emits a ``RepairDecision`` targeting whichever file
    needs to change.

    The ``diagnosis`` field is required — the model writes one sentence
    naming the root-cause file and the underlying mistake before
    committing to a patch. This blocks the failure mode where the
    model "fixes" the test to lower the bar (test gaming) by forcing
    explicit justification first.

    Attributes:
        diagnosis: One sentence on the root cause, naming which file
            (impl or test) carries the bug and why. Cannot be empty.
        impl: The repaired :class:`TaskImplDraft` when the impl needs
            changes, else ``None`` (the on-disk impl is preserved).
        test_source: The repaired pytest module source when the test
            needs changes, else ``None`` (the on-disk test is
            preserved). The source goes through the same syntax +
            evidence gates as the initial test generation.
    """

    model_config = _FROZEN

    diagnosis: str
    impl: TaskImplDraft | None = None
    test_source: str | None = None


class WorkspaceValidation(BaseModel):
    """The verdict of :func:`validate_workspace`.

    Attributes:
        passed: Whether every check passed.
        summary: A human-readable one-line summary.
        failed_checks: Names of the failing checks.
    """

    model_config = _FROZEN

    passed: bool
    summary: str
    failed_checks: tuple[str, ...] = ()


# ── Stage: write the workflow IR ─────────────────────────────────────────


def write_workflow_ir(contract: WorkflowContract, *, layout: MaterializedLayout) -> str:
    """Serialize ``contract`` to ``ir/workflow.yaml`` and return the YAML text."""
    contract_dict = default_compiler.contract_to_dict(contract)
    yaml_text = default_compiler.ir_to_yaml(contract_dict)
    layout.write(layout.workflow_yaml_path(), yaml_text)
    return yaml_text


# ── Stage: compile per-task IR ───────────────────────────────────────────


_TASK_IR_SYSTEM_PROMPT = (
    "Given one TaskIO entry from a workflow contract plus the typed "
    "plan, draft a TaskIRBrief: the task's responsibility, success "
    "criteria, and minimal test expectations. Name only symbols listed "
    "in the plan's PlanStep.api_refs (composition primitives the "
    "planner already grounded). Set is_stub=true when the matching "
    "PlanStep has no api_refs — never invent a symbol to fill the gap."
)


async def compile_task_ir(
    *,
    router: Router,
    contract: WorkflowContract,
    plan_graph: PlanGraph,
    layout: MaterializedLayout,
    tier: ModelTier = ModelTier.DEFAULT,
) -> tuple[TaskIRBrief, ...]:
    """Draft a per-task IR brief for every task in ``contract``.

    Each brief is written to ``ir/tasks/<task_id>.yaml``. An empty
    contract yields an empty tuple (empty contracts are permitted).
    """
    if not contract.task_io:
        return ()

    step_by_id = {step.id: step for step in plan_graph.steps}

    async def _draft(tio: TaskIO) -> TaskIRBrief:
        step = step_by_id.get(tio.task_id)
        step_block = step.model_dump_json(indent=2) if step is not None else "{}"
        user = f"TaskIO:\n{tio.model_dump_json(indent=2)}\n\nPlanStep:\n{step_block}"
        brief = await router.complete_structured(
            tier=tier,
            system=_TASK_IR_SYSTEM_PROMPT,
            user=user,
            schema=TaskIRBrief,
            node_id=f"CompileTaskIR/{tio.task_id}",
        )
        if brief.task_id != tio.task_id:
            brief = brief.model_copy(update={"task_id": tio.task_id})
        return brief

    briefs = tuple(await asyncio.gather(*[_draft(tio) for tio in contract.task_io]))
    for brief in briefs:
        layout.write(
            layout.task_ir_path(brief.task_id),
            yaml.safe_dump(brief.model_dump(mode="json"), sort_keys=False),
        )
    return briefs


# ── Stage: generate the package skeleton ─────────────────────────────────


def generate_workflow_skeleton(contract: WorkflowContract, *, layout: MaterializedLayout) -> str:
    """Emit the templated experiment package skeleton; return workflow.py path.

    Writes ``src/experiment/__init__.py``, ``src/experiment/workflow.py``,
    and ``src/experiment/tasks/__init__.py``; every file is syntax-checked.

    Raises:
        CodegenError: when a generated file does not compile.
    """
    package_init = layout.package_dir() / "__init__.py"
    workflow_py = layout.workflow_py_path()
    tasks_init = layout.tasks_pkg_dir() / "__init__.py"
    workflow_source = render_workflow_module(contract)
    for source, path in (
        (PACKAGE_INIT_BODY, package_init),
        (workflow_source, workflow_py),
        (TASKS_INIT_BODY, tasks_init),
    ):
        try:
            validate_python(source, str(path))
        except SyntaxError as exc:
            raise CodegenError(f"generated skeleton did not compile ({path}): {exc.msg}") from exc
        layout.write(path, source)
    return str(workflow_py)


# ── Stage: generate per-task tests ───────────────────────────────────────


_TASK_TEST_SYSTEM_PROMPT = (
    "Given a TaskIRBrief plus its PlanStep, draft a pytest module that "
    "exercises the task IN ISOLATION. The task is a plain async "
    "function with signature ``async def <task_function>(ctx)``. It "
    "ALWAYS returns a dict whose keys are PlanStep.io.outputs (the "
    "planner's original output names — they may contain dots, hyphens, "
    "etc.). The test imports the function from the generated package, "
    "constructs a fake ``ctx`` (typically ``types.SimpleNamespace`` "
    "with the right ``inputs`` shape), calls the function, and "
    "asserts on the returned dict.\n"
    "\n"
    "``ctx.inputs`` shape — when the task has ONE upstream dep, it's "
    "that upstream's return dict; when MULTIPLE upstreams, it's a "
    "dict keyed by upstream step id. For an isolated test you "
    "construct whatever shape the task body reads (see the "
    "INPUT LOCALS block in the user message — those names tell you "
    "what keys to populate).\n"
    "\n"
    "ASSERTION SCOPE — keep it tight. The impl and test are generated "
    "independently, so any assertion about the *internals* of a "
    "produced value risks an impedance mismatch (formatting, "
    "whitespace, ordering choices the impl might make differently). "
    "Default to SHAPE-LEVEL assertions:\n"
    "  - dict-shape: the returned dict has the expected output key(s);\n"
    "  - type: the value under each key is the expected Python type "
    "(``str``, ``dict``, the relevant class);\n"
    "  - non-emptiness: a string is non-empty, a list has the right "
    "length, an Atomistic has ``n_atoms > 0``.\n"
    "\n"
    "AVOID asserting:\n"
    "  - exact substring content of generated text (scripts, config "
    "files, file paths formatted with extra whitespace, etc.);\n"
    "  - exact numeric equality on values that depend on impl "
    "choices (specific bond counts, specific coordinates, …);\n"
    "  - the order of items in a returned collection unless the "
    "PlanStep explicitly pins it.\n"
    "Content-level invariants belong to integration tests, not the "
    "isolated unit test the debug loop runs in a subprocess.\n"
    "\n"
    "Imports allowed: stdlib + `pytest` + the exact symbols listed in "
    "the ALLOWED PROJECT IMPORTS preamble (the union of every "
    "PlanStep's api_refs, plus re-exports). For synthetic inputs, "
    "build them from literal Python values or from the api_refs "
    "themselves; project symbols not on the allowlist are rejected by "
    "the codegen-evidence gate. When the api_refs alone aren't enough "
    "to materialize a synthetic input, use a hand-rolled stub class "
    "with the minimum attributes the task reads.\n"
    "\n"
    "The test MUST construct its own minimal synthetic inputs — it "
    "must never import, call, or consume the real output of an "
    "upstream task. Honour the brief's test_expectations: a "
    "'synthetic input:' line names an input the test must build "
    "itself; an 'assert:' line names what it must check. If "
    "is_stub=true, emit a single test calling pytest.skip('stub'). "
    "Return the full module source."
)


def _brief_with_test_sketch(brief: TaskIRBrief, step: PlanStep | None) -> TaskIRBrief:
    """Fold a plan step's ``IsolatedTestSketch`` into the brief.

    The step's ``test_sketch`` carries the minimal synthetic inputs the
    isolated test must build for itself and what it should assert; these
    are threaded into ``TaskIRBrief.test_expectations`` (prefixed so the
    test-generation prompt can tell them apart) rather than into a new
    schema. ``step is None`` — no matching plan step — leaves the brief
    unchanged.
    """
    if step is None:
        return brief
    sketch = step.test_sketch
    extra = tuple(f"synthetic input: {item}" for item in sketch.synthetic_inputs) + tuple(
        f"assert: {item}" for item in sketch.assertion_sketch
    )
    if not extra:
        return brief
    return brief.model_copy(update={"test_expectations": brief.test_expectations + extra})


async def generate_task_tests(
    *,
    router: Router,
    briefs: tuple[TaskIRBrief, ...],
    contract: WorkflowContract,
    plan_graph: PlanGraph,
    layout: MaterializedLayout,
    tier: ModelTier = ModelTier.DEFAULT,
) -> tuple[str, ...]:
    """Generate one pytest module per task plus the topology-pin test.

    Each brief is augmented with the matching :class:`PlanStep`'s
    ``test_sketch`` — the isolated test's minimal synthetic inputs and
    assertion sketch — before generation, so the test exercises the task
    in isolation rather than on real upstream output.

    Returns the tuple of written test-file paths.
    """
    io_by_id = {tio.task_id: tio for tio in contract.task_io}
    paths: list[str] = []
    for brief in briefs:
        brief = _brief_with_test_sketch(brief, plan_graph.step_by_id(brief.task_id))
        if brief.is_stub:
            source = render_stub_test(brief.task_id)
        else:
            source = await _generate_one(
                router=router,
                brief=brief,
                tio=io_by_id.get(brief.task_id),
                plan_graph=plan_graph,
                system=_TASK_TEST_SYSTEM_PROMPT,
                node_prefix="GenerateTaskTests",
                tier=tier,
            )
        paths.append(str(layout.write(layout.task_test_path(brief.task_id), source)))

    structure_test = render_workflow_structure_test("ir/workflow.yaml")
    paths.append(
        str(layout.write(layout.tests_dir() / "test_workflow_structure.py", structure_test))
    )
    return tuple(paths)


# ── Stage: generate per-task implementations ─────────────────────────────


_TASK_IMPL_DRAFT_SYSTEM_PROMPT = (
    "You implement one workflow task. The codegen layer assembles "
    "the module around your draft from the PlanStep included in the "
    "user message — the function name, docstring, input bindings, "
    "and return statement are all already wired. You contribute only "
    "imports and the body:\n"
    "\n"
    '    """<PlanStep.composition_notes>"""\n'
    "    <your imports>\n"
    "\n"
    "    async def <task_function>(ctx):\n"
    "        <local> = ctx.inputs[...]   # auto from each "
    "PlanStep.io.inputs entry\n"
    "        <YOUR BODY>\n"
    "        return {<key>: <local>, ...}  # auto from PlanStep.io.outputs\n"
    "\n"
    "Names in PlanStep.io are free-form (they can be filenames, "
    "domain terms, anything readable); the assembler sanitises them "
    "to Python identifiers when generating the local variables. For "
    "every input/output the EXPECTED LOCAL NAME is listed in the "
    "user message — bind those names in your body.\n"
    "\n"
    "Emit a TaskImplDraft:\n"
    "  - `imports`: each entry is one `import X` or "
    "`from X import Y`. Draw only from symbols listed in "
    "PlanStep.api_refs (or re-exports of the same symbol at a "
    "shorter path).\n"
    "  - `body`: Python statements that compose the imports to bind "
    "the expected output locals. No leading indentation; the "
    "assembler indents.\n"
    "  - `is_stub`: true when the PlanStep has no api_refs (a "
    "placeholder); false otherwise."
)


async def generate_task_implementations(
    *,
    router: Router,
    briefs: tuple[TaskIRBrief, ...],
    contract: WorkflowContract,
    plan_graph: PlanGraph,
    layout: MaterializedLayout,
    tier: ModelTier = ModelTier.DEFAULT,
) -> tuple[str, ...]:
    """Generate one runnable module per task. Returns the written paths."""
    del contract  # io is derived from PlanStep now; TaskIO is informational only
    paths: list[str] = []
    for brief in briefs:
        step = plan_graph.step_by_id(brief.task_id)
        if step is None:
            # Plan has no matching step — treat as a stub so the
            # generated workspace still loads even if the IR is
            # incomplete.
            paths.append(str(layout.write(layout.task_impl_path(brief.task_id), "")))
            continue
        if brief.is_stub:
            source = assemble_impl_module(TaskImplDraft(is_stub=True), step)
        else:
            source = await _generate_impl_one(
                router=router,
                brief=brief,
                step=step,
                plan_graph=plan_graph,
                tier=tier,
            )
        paths.append(str(layout.write(layout.task_impl_path(brief.task_id), source)))
    return tuple(paths)


def assemble_impl_module(draft: TaskImplDraft, step: PlanStep) -> str:
    """Render a :class:`TaskImplDraft` into a full Python module source.

    Everything structural — module docstring, function name, input
    bindings, return shape — is derived from ``step``. The model
    contributes ONLY ``draft.imports`` and ``draft.body``. The
    assembled shape is::

        \"\"\"<step.composition_notes>\"\"\"

        <draft.imports>


        async def <module_id(step.id)>(ctx):
            <local> = ctx.inputs[...]   # one per PlanStep.io.inputs
            <draft.body>                # indented
            return {<key>: <local>, ...}  # one per PlanStep.io.outputs

    Input / output names in the PlanGraph are free-form (the planner
    may write ``data.peo``, ``peo_chain``, ``with-hyphen``, …); the
    assembler routes them through :func:`module_id` to produce a
    valid Python identifier for the local-variable binding, while
    keeping the original name as the dict KEY so producer / consumer
    steps still rendezvous under the planner's chosen label.

    For a stub draft the body becomes ``raise NotImplementedError``
    and the imports / return are dropped — a stub must always load,
    even when an api_ref is unavailable in the host env.
    """
    import textwrap

    func_name = module_id(step.id)
    docstring = _escape_triple_quoted(step.composition_notes.strip())

    sections: list[str] = []
    if docstring:
        sections.append(f'"""{docstring}"""')

    # Imports are dropped on stubs so a missing api_ref doesn't break
    # module loading.
    if not draft.is_stub and draft.imports:
        sections.append("\n".join(line.rstrip() for line in draft.imports))

    func_lines = [f"async def {func_name}(ctx):"]
    if draft.is_stub or not draft.body.strip():
        # repr() yields a correctly-escaped Python literal regardless
        # of what step.id contains (quotes, backslashes, non-ASCII).
        func_lines.append(f"    raise NotImplementedError({step.id + ' is a stub'!r})")
    else:
        # Auto-bind named inputs from upstream return values. The
        # workflow runtime delivers ``ctx.inputs`` as the upstream's
        # return value when there's exactly one dep, or as a dict keyed
        # by upstream-step id when there are multiple. We materialise
        # both into ``<local> = ctx.inputs[...]`` lines so the body
        # sees plain local variables and can never get the shape wrong.
        dep_count = len(step.depends_on)
        for inp in step.io.inputs:
            if inp.source_step is None:
                # Externally supplied; bind via config or treat as kwarg
                # in user body. Leave it to the body to read.
                continue
            local = module_id(inp.name)
            if dep_count <= 1:
                func_lines.append(f"    {local} = ctx.inputs[{inp.name!r}]")
            else:
                func_lines.append(f"    {local} = ctx.inputs[{inp.source_step!r}][{inp.name!r}]")
        # User body — indented under the function.
        func_lines.append(textwrap.indent(draft.body.strip("\n"), "    "))
        # Auto return — from PlanStep.io.outputs. Tasks always return a
        # dict keyed by the planner's original name; the local variable
        # the body binds is the sanitised form (so the dict key can be
        # ``data.peo`` while the local is ``data_peo``).
        if step.io.outputs:
            return_pairs = ", ".join(f"{name!r}: {module_id(name)}" for name in step.io.outputs)
            func_lines.append(f"    return {{{return_pairs}}}")
        else:
            func_lines.append("    return None")
    sections.append("\n".join(func_lines))

    return "\n\n\n".join(sections) + "\n"


def _escape_triple_quoted(text: str) -> str:
    """Make ``text`` safe to embed inside a ``\\\"\\\"\\\" … \\\"\\\"\\\"`` literal.

    The naive ``f'\\\"\\\"\\\"{text}\\\"\\\"\\\"'`` interpolation breaks when
    ``text`` itself contains ``\\\"\\\"\\\"`` (LLM docstrings often quote
    examples). Replacing the embedded triple-quote with a single-quote
    triplet keeps the docstring renderable while preserving the user-
    visible text. Backslashes are also escaped so ``\\u`` and friends
    in the docstring don't get interpreted by the Python tokenizer.
    """
    if not text:
        return text
    return text.replace("\\", "\\\\").replace('"""', "'''")


async def _generate_impl_one(
    *,
    router: Router,
    brief: TaskIRBrief,
    step: PlanStep,
    plan_graph: PlanGraph,
    tier: ModelTier,
) -> str:
    """Generate, validate, and assemble one task implementation module.

    The model emits a :class:`TaskImplDraft` (imports + body); the
    assembler wraps it in the plain async-function shape derived from
    ``step`` (function name, docstring, input bindings, return). The
    gate runs on the assembled source; on failure the verdict is fed
    back into the next prompt up to :data:`_GATE_RETRY_BUDGET` attempts.
    """
    allowed_refs = sorted({ref for s in plan_graph.steps for ref in s.api_refs})
    refs_block = "\n".join(f"  - {ref}" for ref in allowed_refs) or "  (none)"
    # Surface the sanitised local names explicitly so the body can bind
    # them directly — the LLM doesn't need to compute the mapping.
    input_locals = [
        f"  - {module_id(inp.name)}  (from PlanStep.io.inputs[{i}], original name {inp.name!r})"
        for i, inp in enumerate(step.io.inputs)
        if inp.source_step is not None
    ]
    output_locals = [
        f"  - {module_id(name)}  (returned under key {name!r})" for name in step.io.outputs
    ]
    bindings_block = (
        "INPUT LOCALS — already bound for you before your body runs:\n"
        + ("\n".join(input_locals) if input_locals else "  (none — root task)")
        + "\n\nOUTPUT LOCALS — your body must bind each of these:\n"
        + ("\n".join(output_locals) if output_locals else "  (none — return None)")
    )
    base_user = (
        "ALLOWED PROJECT IMPORTS — every entry in `imports` MUST "
        "draw from this exact list (re-exports of the same symbol at a "
        "shorter path are permitted; nothing else is). The codegen "
        "evidence gate runs on the assembled module and will reject "
        "anything outside this list.\n"
        f"{refs_block}\n\n"
        f"{bindings_block}\n\n"
        f"TaskIRBrief:\n{brief.model_dump_json(indent=2)}\n\n"
        f"PlanStep:\n{step.model_dump_json(indent=2)}"
    )
    where = f"GenerateTaskImplementations/{brief.task_id}"

    user = base_user
    last_issue: str | None = None
    last_source: str | None = None
    for _attempt in range(_GATE_RETRY_BUDGET):
        if last_issue is not None:
            user = (
                f"{base_user}\n\n"
                f"PREVIOUS ATTEMPT WAS REJECTED by the codegen gate:\n"
                f"  {last_issue}\n\n"
                "Re-emit the TaskImplDraft fixing that issue. The "
                "ALLOWED PROJECT IMPORTS list still applies."
            )
        draft = await router.complete_structured(
            tier=tier,
            system=_TASK_IMPL_DRAFT_SYSTEM_PROMPT,
            user=user,
            schema=TaskImplDraft,
            node_id=where,
        )
        source = assemble_impl_module(draft, step)
        issue = validate_assembled_impl(source, plan_graph)
        if issue is not None:
            last_issue = issue
            last_source = source
            continue
        return source

    # Budget exhausted — preserve the structured missing payload when
    # the failure was an evidence-gate rejection so AuthorMode's repair-
    # diff pipeline (``materialize._diffs_from_codegen_error``) can emit
    # a ``repair_proposed`` event for the caller. Without this the diff
    # is empty and the failure surfaces with no audit trail.
    assert last_issue is not None
    if "un-evidenced symbols" in last_issue and last_source is not None:
        re_missing = validate_codegen_evidence(last_source, plan_graph)
        raise CodegenError(f"{where}: {last_issue}", missing=re_missing)
    raise CodegenError(f"{where}: {last_issue}")


async def _generate_one(
    *,
    router: Router,
    brief: TaskIRBrief,
    tio: TaskIO | None,
    plan_graph: PlanGraph,
    system: str,
    node_prefix: str,
    tier: ModelTier,
) -> str:
    """Generate, syntax-check, and evidence-gate one test module's source.

    Used by :func:`generate_task_tests`. Impl generation goes through
    :func:`_generate_impl_one` (which uses the constrained
    :class:`TaskImplDraft` schema).
    """
    del tio  # task-IR carries everything the test needs from the plan
    step = plan_graph.step_by_id(brief.task_id)
    step_block = step.model_dump_json(indent=2) if step is not None else "{}"
    allowed_refs = sorted({ref for s in plan_graph.steps for ref in s.api_refs})
    refs_block = "\n".join(f"  - {ref}" for ref in allowed_refs) or "  (none)"
    allowed_namespaces = sorted({ref.split(".", 1)[0] for ref in allowed_refs if "." in ref})
    namespaces_phrase = (
        " / ".join(f"`{ns}`" for ns in allowed_namespaces) if allowed_namespaces else "any"
    )
    # Surface the sanitised local-variable names so the test knows what
    # ``ctx.inputs`` keys to populate and how to read the task's return.
    func_name = module_id(brief.task_id) if step is None else module_id(step.id)
    input_locals = (
        [
            f"  - {module_id(inp.name)}  (key {inp.name!r} in ctx.inputs)"
            for inp in step.io.inputs
            if inp.source_step is not None
        ]
        if step is not None
        else []
    )
    output_keys = (
        [f"  - {name!r}  (sanitised local: {module_id(name)})" for name in step.io.outputs]
        if step is not None
        else []
    )
    bindings_block = (
        f"TASK FUNCTION: ``async def {func_name}(ctx)``\n"
        f"IMPORT THE TASK FUNCTION VIA:\n"
        f"  ``from experiment.tasks.{func_name} import {func_name}``\n"
        f"(this is the only correct import path — the test runs with "
        f"`src/` on PYTHONPATH, so the impl is reachable as "
        f"``experiment.tasks.{func_name}``; never use a bare "
        f"``from {func_name} import …`` or `sys.path` hacks.)\n\n"
        "INPUT KEYS — what your synthetic ctx.inputs should contain:\n"
        + ("\n".join(input_locals) if input_locals else "  (none — root task; ctx.inputs is None)")
        + "\n\nOUTPUT KEYS — the returned dict's keys (assert on these):\n"
        + ("\n".join(output_keys) if output_keys else "  (none — task returns None)")
    )
    base_user = (
        f"ALLOWED PROJECT IMPORTS — any reference to a {namespaces_phrase} "
        "symbol MUST be drawn from this exact list (re-exports of the "
        "same symbol at a shorter path are permitted; nothing else is). "
        "Importing or referencing a non-listed symbol will fail the "
        "codegen-evidence gate and the whole module will be rejected.\n"
        f"{refs_block}\n\n"
        f"{bindings_block}\n\n"
        f"TaskIRBrief:\n{brief.model_dump_json(indent=2)}\n\n"
        f"PlanStep:\n{step_block}"
    )
    where = f"{node_prefix}/{brief.task_id}"

    # Inline retry — when a per-task gate (shape / evidence / syntax)
    # rejects the first emission, append the gate's verdict to the next
    # prompt and ask the model to fix it. Two retries (3 total attempts)
    # is enough to recover from a one-off prompt slip without burning a
    # full ResearchAndPlan rewind.
    user = base_user
    last_issue: str | None = None
    last_source: str | None = None
    for _attempt in range(_GATE_RETRY_BUDGET):
        if last_issue is not None:
            user = (
                f"{base_user}\n\n"
                f"PREVIOUS ATTEMPT WAS REJECTED by the codegen gate:\n"
                f"  {last_issue}\n\n"
                "Re-emit the module FIXING that specific issue. The "
                "constraints in the system prompt and the ALLOWED "
                "PROJECT IMPORTS block still hold."
            )
        module = await router.complete_structured(
            tier=tier,
            system=system,
            user=user,
            schema=GeneratedModule,
            node_id=f"{node_prefix}/{brief.task_id}",
        )
        try:
            validate_python(module.source, where)
        except SyntaxError as exc:
            last_issue = f"generated module did not compile: {exc.msg}"
            last_source = module.source
            continue
        missing = validate_codegen_evidence(module.source, plan_graph)
        if missing:
            refs = ", ".join(m.ref for m in missing)
            last_issue = f"generated module references un-evidenced symbols: {refs}"
            last_source = module.source
            continue
        return module.source

    # Budget exhausted — raise the last gate verdict so the caller's
    # ``CodegenError`` handling kicks in.
    assert last_issue is not None
    if "un-evidenced symbols" in last_issue:
        # Preserve the structured ``missing`` payload for the caller.
        re_missing = validate_codegen_evidence(last_source, plan_graph) if last_source else ()
        raise CodegenError(f"{where}: {last_issue}", missing=re_missing)
    raise CodegenError(f"{where}: {last_issue}")


def validate_assembled_impl(source: str, plan_graph: PlanGraph) -> str | None:
    """Run the three impl-codegen gates over an assembled module.

    Returns ``None`` when the source passes all gates; otherwise a
    human-readable issue suitable for feeding back into the next prompt.
    Wraps :func:`validate_python` + :func:`_check_impl_shape` +
    :func:`validate_codegen_evidence` so every code path that produces an
    impl module (initial codegen + debug-loop repair) gates identically.
    """
    try:
        validate_python(source, "<assembled_impl>")
    except SyntaxError as exc:
        return f"assembled module did not compile: {exc.msg}"
    shape_issue = _check_impl_shape(source)
    if shape_issue is not None:
        return shape_issue
    missing = validate_codegen_evidence(source, plan_graph)
    if missing:
        refs = ", ".join(m.ref for m in missing)
        return f"references un-evidenced symbols: {refs}"
    return None


def validate_test_source(source: str, plan_graph: PlanGraph) -> str | None:
    """Run the test-codegen gates over a generated test module.

    Tests don't have a shape gate (they're free-form pytest modules
    with fixtures, classes, multiple functions); the gate is syntax
    + evidence. Returns ``None`` on success, else a human-readable
    issue ready for retry-prompt feedback.
    """
    try:
        validate_python(source, "<test>")
    except SyntaxError as exc:
        return f"test module did not compile: {exc.msg}"
    missing = validate_codegen_evidence(source, plan_graph)
    if missing:
        refs = ", ".join(m.ref for m in missing)
        return f"test references un-evidenced symbols: {refs}"
    return None


def _check_impl_shape(source: str) -> str | None:
    """Return a human-readable issue if ``source`` doesn't fit the function
    impl shape.

    The new codegen output is a plain async-function module: module
    docstring (optional) + imports + exactly one ``async def`` taking
    ``ctx``. Anything else — a class, a top-level assignment, an extra
    function, a test definition — is rejected. This removes the surface
    for module-level monkey-patching and stray helpers entirely.
    """
    tree = ast.parse(source)

    task_funcs: list[str] = []
    bad_top_level: list[str] = []

    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            # Module docstring.
            continue
        if isinstance(stmt, ast.Import | ast.ImportFrom):
            continue
        if isinstance(stmt, ast.AsyncFunctionDef):
            task_funcs.append(stmt.name)
            args = [a.arg for a in stmt.args.args]
            if args != ["ctx"]:
                return (
                    f"`async def {stmt.name}({', '.join(args)})` is the wrong "
                    "signature — the task function must accept exactly one "
                    "parameter named `ctx`"
                )
            if stmt.name.startswith("test_"):
                return (
                    f"top-level `async def {stmt.name}` looks like a test; "
                    "the test file is generated separately, the impl module "
                    "must hold only the task function"
                )
            continue
        # Anything else at module level — class, def, assignment — is banned.
        if isinstance(stmt, ast.ClassDef):
            bad_top_level.append(f"class {stmt.name}")
        elif isinstance(stmt, ast.FunctionDef):
            bad_top_level.append(f"def {stmt.name}")
        elif isinstance(stmt, ast.Assign):
            target_repr = ast.unparse(stmt.targets[0]) if stmt.targets else "<assign>"
            bad_top_level.append(f"`{target_repr} = ...`")
        elif isinstance(stmt, ast.AnnAssign):
            target_repr = ast.unparse(stmt.target)
            bad_top_level.append(f"`{target_repr} = ...`")
        else:
            bad_top_level.append(type(stmt).__name__)

    if bad_top_level:
        return (
            "implementation module has top-level statements outside the "
            "allowed shape (module docstring + imports + one `async def "
            f"<task>(ctx)`): {', '.join(bad_top_level)}. The assembler "
            "generates the function wrapper; the model contributes only "
            "the imports and the function body."
        )
    if not task_funcs:
        return (
            "implementation module has no `async def <task>(ctx)` function "
            "at module level — the assembled module must define exactly one"
        )
    if len(task_funcs) > 1:
        return (
            "implementation module defines multiple top-level async "
            f"functions ({', '.join(task_funcs)}); exactly one is allowed"
        )
    return None


# ── Stage: validate the materialized workspace ───────────────────────────


def validate_workspace(contract: WorkflowContract) -> WorkspaceValidation:
    """Run :func:`~molexp.workflow.validate_workflow_contract` over the IR.

    The materialized IR is the source of truth; this re-runs the
    workflow-layer contract checks and folds the report into a
    :class:`WorkspaceValidation` verdict.
    """
    report = validate_workflow_contract(contract)
    failed = tuple(issue.check_id.value for issue in report.issues if issue.severity == "error")
    if report.ok:
        summary = f"workspace validation passed ({len(contract.task_io)} task(s))"
    else:
        summary = f"workspace validation failed: {len(failed)} error(s)"
    return WorkspaceValidation(passed=report.ok, summary=summary, failed_checks=failed)
