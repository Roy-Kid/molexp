"""Workflow compiler — convert between IR, Python script, Mermaid, and Spec.

The workflow has four equivalent surfaces:

- :class:`Workflow` — in-memory pydantic-graph compiled object (the
  execution entry point).
- **IR** — JSON-serializable ``dict`` matching ``schema/workflow.json``.
  This is the wire format used by the server and the agent.
- **Python script** — a runnable molexp module that assigns the IR to a
  top-level ``WORKFLOW_IR`` literal followed by
  ``Workflow.from_dict(WORKFLOW_IR)``. Editable by humans; the IR
  embedded in the script is the source of truth.
- **Mermaid** — a ``flowchart LR`` text rendering for read-only review
  surfaces.

:class:`WorkflowCompiler` provides pairwise converters between these
surfaces and round-trip guarantees where they apply::

    ir = compiler.python_to_ir(compiler.ir_to_python(ir))  # exact
    ir = compiler.spec_to_ir(compiler.ir_to_spec(ir))  # exact (slugged tasks)
    text = compiler.ir_to_mermaid(ir)  # one-way

Every conversion is **AST-based / template-based** — we never ``exec``
user-supplied Python. The Python surface is a structured carrier for
the IR, not a free-form script.

PlanMode-specific report digestion, code generation, and handoff
assembly are not part of this module. They belong to the agent layer;
the workflow layer only owns generic workflow representations and
contract validation.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import yaml

from .._typing import JSONMapping, JSONValue
from .contract import WorkflowContract
from .spec import Workflow

if TYPE_CHECKING:
    from .registry import TaskTypeRegistry

__all__ = [
    "WorkflowCompiler",
    "default_compiler",
]


_SCRIPT_HEADER = '''\
"""Auto-generated molexp workflow plan.

The IR literal below is the source of truth — edit it, and the
matching ``Workflow.from_dict(WORKFLOW_IR)`` call below will
load your edits. The server reads the IR (not this script) when
running the workflow; the script exists so the IR is comfortable
to read and review in Python form.
"""
from molexp.workflow.spec import Workflow
'''


class WorkflowCompiler:
    """Stateless converter between workflow representations.

    Methods are instance methods (rather than classmethods) so callers
    can subclass and override individual conversions — e.g. swap the
    Mermaid renderer for a Graphviz one — without touching the rest of
    the API. The default instance is also exposed at module level as
    :data:`default_compiler` for callers that don't need customization.
    """

    # ── IR ↔ Python script ─────────────────────────────────────────────

    def ir_to_python(self, ir: JSONMapping) -> str:
        """Render a workflow IR dict as a runnable Python script.

        The script assigns the IR to a top-level ``WORKFLOW_IR`` literal
        and then loads it via :meth:`Workflow.from_dict`. Re-parsing
        the script with :meth:`python_to_ir` yields back the same IR
        (modulo dict ordering and Python repr quirks of
        :func:`ast.literal_eval`).
        """
        if not isinstance(ir, dict):
            raise ValueError("ir_to_python: ir must be a dict.")
        # ``ir`` is a Mapping; ``_safe_literal_repr`` accepts ``JSONValue``
        # which includes the concrete ``dict[str, JSONValue]`` arm. Convert
        # so the static-typing narrowing matches the dict arm explicitly.
        literal = _safe_literal_repr(dict(ir))
        return (
            f"{_SCRIPT_HEADER}\nWORKFLOW_IR = {literal}\n\nspec = Workflow.from_dict(WORKFLOW_IR)\n"
        )

    def python_to_ir(self, script: str) -> JSONMapping:
        """Extract the IR literal from a Python script.

        Walks the module AST looking for a top-level
        ``WORKFLOW_IR = <dict literal>`` and decodes it via
        :func:`ast.literal_eval`. Anything else in the script (imports,
        comments, the trailing ``Workflow.from_dict`` call) is
        ignored — we never execute user-supplied code.
        """
        try:
            tree = ast.parse(script)
        except SyntaxError as exc:
            raise ValueError(f"python_to_ir: invalid Python: {exc}") from exc
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            targets = node.targets
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                continue
            if targets[0].id != "WORKFLOW_IR":
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"python_to_ir: WORKFLOW_IR is not a literal: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError("python_to_ir: WORKFLOW_IR must be a dict literal.")
            return value
        raise ValueError("python_to_ir: no top-level WORKFLOW_IR assignment found.")

    # ── IR ↔ Workflow ───────────────────────────────────────────────

    def ir_to_spec(self, ir: JSONMapping, *, registry: TaskTypeRegistry | None = None) -> Workflow:
        """Build a :class:`Workflow` from JSON IR.

        Thin wrapper over :meth:`Workflow.from_dict`; exposed here
        so callers have one obvious entry point for *any* representation
        conversion.
        """
        return Workflow.from_dict(ir, registry=registry)

    def spec_to_ir(self, spec: Workflow) -> JSONMapping:
        """Serialize a :class:`Workflow` to JSON IR."""
        return spec.to_dict()

    # ── Spec → Python / Mermaid (composition) ───────────────────────────

    def spec_to_python(self, spec: Workflow) -> str:
        """Render a spec as a Python script (via the IR)."""
        return self.ir_to_python(self.spec_to_ir(spec))

    def spec_to_mermaid(self, spec: Workflow) -> str:
        """Render a spec as a Mermaid flowchart (via the IR)."""
        return self.ir_to_mermaid(self.spec_to_ir(spec))

    # ── WorkflowContract ↔ dict ─────────────────────────────────────────

    def contract_to_dict(self, contract: WorkflowContract) -> JSONMapping:
        """Serialize a :class:`WorkflowContract` to a JSON-shaped dict.

        Uses pydantic's ``mode="json"`` dump so enums become strings
        and nested models become plain dicts; the result is safe for
        ``yaml.safe_dump`` and ``json.dumps``.
        """
        return contract.model_dump(mode="json")

    def dict_to_contract(self, data: JSONMapping) -> WorkflowContract:
        """Build a :class:`WorkflowContract` from its dict form.

        Mirror of :meth:`contract_to_dict`. Validation runs through
        pydantic — unknown keys trip ``extra="forbid"``.
        """
        return WorkflowContract.model_validate(data)

    # ── IR ↔ YAML ───────────────────────────────────────────────────────
    #
    # YAML is offered for the experiment-workspace ``ir/workflow.yaml``
    # use case PlanMode produces. JSON remains the canonical wire format
    # for the server. ``safe_load`` / ``safe_dump`` only — untrusted YAML
    # is never executed.

    def ir_to_yaml(self, ir: JSONMapping) -> str:
        """Render any IR-shaped dict as YAML text via ``yaml.safe_dump``."""
        if not isinstance(ir, dict):
            raise ValueError("ir_to_yaml: ir must be a dict.")
        return yaml.safe_dump(dict(ir), sort_keys=False, default_flow_style=False)

    def yaml_to_ir(self, text: str) -> JSONMapping:
        """Parse YAML text into an IR-shaped dict via ``yaml.safe_load``.

        Rejects YAML whose top-level value is not a mapping; this
        preserves the IR contract that workflow / contract dicts are
        always object-shaped at the root. Unsafe tags
        (``!!python/object/...``) raise the safe-loader's
        :class:`yaml.constructor.ConstructorError` (a subclass of
        :class:`yaml.YAMLError`) — no Python object is constructed.
        """
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"yaml_to_ir: top-level YAML value must be a mapping, got {type(loaded).__name__}."
            )
        return loaded

    # ── Spec ↔ YAML (composition) ───────────────────────────────────────

    def spec_to_yaml(self, spec: Workflow) -> str:
        """Render a spec as YAML text (via the JSON IR)."""
        return self.ir_to_yaml(self.spec_to_ir(spec))

    def yaml_to_spec(self, text: str, *, registry: TaskTypeRegistry | None = None) -> Workflow:
        """Build a :class:`Workflow` from a YAML text (via the JSON IR)."""
        return self.ir_to_spec(self.yaml_to_ir(text), registry=registry)

    # ── IR → Mermaid (one-way) ──────────────────────────────────────────

    def ir_to_mermaid(self, ir: JSONMapping) -> str:
        """Render a workflow IR as a ``flowchart LR`` Mermaid block.

        The renderer emits one node per ``task_configs`` entry (label =
        ``task_id`` + ``task_type``) and one edge per ``links`` entry.
        Tasks with no incoming or outgoing edge still appear as
        standalone nodes so the diagram surfaces orphans.
        """
        if not isinstance(ir, dict):
            raise ValueError("ir_to_mermaid: ir must be a dict.")
        task_configs = _as_object_list(ir.get("task_configs"))
        links = _as_object_list(ir.get("links"))

        lines = ["flowchart LR"]
        for tc in task_configs:
            tid = _mermaid_id(tc.get("task_id", "?"))
            ttype_raw = tc.get("task_type", "")
            ttype = ttype_raw if isinstance(ttype_raw, str) else ""
            task_id_text = str(tc.get("task_id", "?"))
            label = f"{task_id_text}<br/><i>{ttype}</i>" if ttype else task_id_text
            lines.append(f'  {tid}["{label}"]')
        for link in links:
            src = _mermaid_id(link.get("source", "?"))
            tgt = _mermaid_id(link.get("target", "?"))
            lines.append(f"  {src} --> {tgt}")
        return "\n".join(lines) + "\n"


default_compiler = WorkflowCompiler()


# ── Helpers ────────────────────────────────────────────────────────────────


def _safe_literal_repr(value: JSONValue, level: int = 0) -> str:
    """Pretty-print a literal-safe value with two-space indentation per level.

    Reuses :func:`repr` for scalars and recurses into lists / dicts so
    the output is one entry per line — diff-reviewable. Rejects non-
    literal values up front so the caller fails before producing a
    script that won't round-trip.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, dict):
        if not value:
            return "{}"
        pad = "    " * (level + 1)
        close_pad = "    " * level
        lines = [f"{pad}{k!r}: {_safe_literal_repr(v, level + 1)}" for k, v in value.items()]
        body = ",\n".join(lines)
        return "{\n" + body + ",\n" + close_pad + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        pad = "    " * (level + 1)
        close_pad = "    " * level
        lines = [f"{pad}{_safe_literal_repr(v, level + 1)}" for v in value]
        body = ",\n".join(lines)
        return "[\n" + body + ",\n" + close_pad + "]"
    if isinstance(value, tuple):
        return _safe_literal_repr(list(value), level)
    raise ValueError(
        f"ir_to_python: value of type {type(value).__name__!r} is not "
        "literal-safe; the IR must be JSON-compatible scalars / lists / dicts."
    )


def _mermaid_id(raw: JSONValue) -> str:
    """Coerce a task_id into a Mermaid-safe identifier."""
    s = str(raw)
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in s)
    return f"n_{safe}" if safe else "n_unnamed"


def _as_object_list(value: JSONValue | None) -> list[dict[str, JSONValue]]:
    """Narrow a ``JSONValue`` field expected to hold a list of JSON objects.

    Accepts ``None`` / non-list / list-of-non-dict by returning an empty
    list, since the IR Mermaid renderer treats those as "no entries"
    rather than an error condition.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
