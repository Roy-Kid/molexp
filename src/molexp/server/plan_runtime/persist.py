"""Persist a plan-generated workflow onto its Experiment for UI display.

PlanMode's ``workflow_source`` artifact is generated Python (a ``build_workflow()``
program); the UI workflow-graph renderer reads ``experiment.workflow_source`` as
a ``molexp.workflow`` IR document (the shape the ``workflow`` route round-trips
through ``default_codec``). This compiles the already-pipeline-validated source
into a ``CompiledWorkflow`` and serializes it via ``default_codec.spec_to_ir`` —
then saves it on the experiment so the existing renderer picks it up.

The source was already compiled by ``ValidateWorkflowSource`` in the pipeline
(same restricted-builtins ``exec``), so re-compiling here introduces no new trust
boundary; a compile failure is logged and skipped, never raised.
"""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger

if TYPE_CHECKING:
    from molexp._typing import JSONValue
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.run import Run

__all__ = ["persist_plan_workflow_to_experiment"]

_LOG = get_logger(__name__)

# Minimal builtins the generated program may use — mirrors the allow-list in
# ``harness/stages/validate_workflow_source.py`` (not the real ``builtins``).
_SAFE_BUILTINS: dict[str, Any] = {
    "__import__": __import__,
    "len": len,
    "range": range,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "sum": sum,
    "min": min,
    "max": max,
}


def _attach_task_sources(ir: dict[str, JSONValue], source: str) -> None:
    """Annotate each ``task_config`` with its own source code (in place).

    The generated program defines one ``@wf.task``-decorated function per task,
    its name equal to the task id. We AST-split *source*, slice each matching
    function (decorators included), dedent it, and stash it on the node as
    ``source`` so the graph inspector can show exactly what a node runs. A
    parse failure or a name with no function is skipped silently — the source
    annotation is observability sugar, never load-bearing.
    """
    task_configs = ir.get("task_configs")
    if not isinstance(task_configs, list):
        return
    wanted = {
        tc["task_id"]
        for tc in task_configs
        if isinstance(tc, dict) and isinstance(tc.get("task_id"), str)
    }
    if not wanted:
        return
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    lines = source.splitlines()
    by_name: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in wanted or node.end_lineno is None:
            continue
        # Span from the first decorator (or the def line) through the body end,
        # then dedent away the nesting inside ``build_workflow``.
        start = min([d.lineno for d in node.decorator_list] + [node.lineno])
        segment = "\n".join(lines[start - 1 : node.end_lineno])
        by_name[node.name] = textwrap.dedent(segment).strip("\n")
    for tc in task_configs:
        if isinstance(tc, dict) and tc.get("task_id") in by_name:
            tc["source"] = by_name[tc["task_id"]]


def _annotation_to_ui_type(ann: ast.expr | None) -> tuple[str, list | None]:
    """Map a parameter annotation to a UI field type (+ enum options)."""
    if isinstance(ann, ast.Name):
        return {"float": "number", "int": "integer", "str": "text", "bool": "boolean"}.get(
            ann.id, "text"
        ), None
    if isinstance(ann, ast.Subscript):
        base = ann.value
        base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
        if base_name == "Literal":
            elts = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
            options = [e.value for e in elts if isinstance(e, ast.Constant)]
            return "enum", options
    return "text", None


def _extract_input_schema(ir: dict[str, JSONValue], source: str) -> None:
    """Derive the workflow's editable inputs from the tasks' typed parameters.

    A task parameter that carries a DEFAULT is a configuration knob the
    scientist sets (``sigma: float = 1.0``); dataflow parameters (bound from
    upstream outputs) have no default and are skipped. The deduped union becomes
    ``ir["input_schema"]`` — ``[{name, type, default, options?}]`` — which the UI
    renders as a typed form (number / text / boolean / enum-dropdown).
    """
    task_configs = ir.get("task_configs")
    if not isinstance(task_configs, list):
        return
    wanted = {
        tc["task_id"]
        for tc in task_configs
        if isinstance(tc, dict) and isinstance(tc.get("task_id"), str)
    }
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    fields: dict[str, dict] = {}

    def _record(arg: ast.arg, default: ast.expr | None) -> None:
        name = arg.arg
        if default is None or name in ("ctx", "self") or name in fields:
            return
        ftype, options = _annotation_to_ui_type(arg.annotation)
        try:
            default_value = ast.literal_eval(default)
        except (ValueError, SyntaxError):
            default_value = None
        field: dict = {"name": name, "type": ftype, "default": default_value}
        if options is not None:
            field["options"] = options
        fields[name] = field

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in wanted:
            continue
        pos = node.args.args
        for arg, default in zip(
            pos[len(pos) - len(node.args.defaults) :], node.args.defaults, strict=False
        ):
            _record(arg, default)
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=False):
            _record(arg, default)

    if fields:
        ir["input_schema"] = list(fields.values())


def _compile_source_to_ir(source: str) -> dict[str, JSONValue] | None:
    """Compile a ``build_workflow()`` program to a UI-renderable IR document."""
    import molexp.workflow as workflow

    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(source, "<plan_workflow_source>", "exec"), namespace)
        builder = namespace["build_workflow"]()
        compiled = builder.compile()
        # strict=False: this is observability (the displayed graph), not a
        # round-trip — slug-less tasks serialize as task_type=None rather than
        # raising, mirroring the live workflow.json graph the UI already renders.
        ir = dict(workflow.default_codec.spec_to_ir(compiled, strict=False))
        _attach_task_sources(ir, source)
        _extract_input_schema(ir, source)
        return ir
    except Exception as exc:
        _LOG.warning(f"plan workflow source did not compile for display: {exc!r}")
        return None


def persist_plan_workflow_to_experiment(run: Run, experiment: Experiment) -> bool:
    """Compile the run's ``workflow_source`` artifact to IR and save it on ``experiment``.

    Returns ``True`` when an IR document was persisted; ``False`` when no
    ``workflow_source`` artifact exists or it failed to compile (logged).
    """
    from molexp.harness.schemas import WorkflowSource
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))
    ref = store.latest_by_kind("workflow_source")
    if ref is None:
        return False
    ws = WorkflowSource.model_validate_json(store.get(ref.id))
    ir = _compile_source_to_ir(ws.source)
    if ir is None:
        return False
    experiment.metadata = experiment.metadata.model_copy(
        update={"workflow_source": json.dumps(ir, sort_keys=True)}
    )
    experiment.save()
    return True
