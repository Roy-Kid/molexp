"""Workflow compiler — convert between IR, Python script, Mermaid, and Spec.

The workflow has four equivalent surfaces:

- :class:`WorkflowSpec` — in-memory pydantic-graph compiled object (the
  execution entry point).
- **IR** — JSON-serializable ``dict`` matching ``schema/workflow.json``.
  This is the wire format used by the server and the agent.
- **Python script** — a runnable molexp module that assigns the IR to a
  top-level ``WORKFLOW_IR`` literal followed by
  ``WorkflowSpec.from_dict(WORKFLOW_IR)``. Editable by humans; the IR
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

Plan-side compilation (``PlanProposal`` → ``ParameterizedWorkflowSpec``
or → ``WorkflowSpec``) is not part of this module. It belongs to the
agent layer and lives in ``molexp.agent.planning.proposal_compiler``;
the workflow layer is unaware of plans.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .._typing import JSONMapping, JSONValue
from .spec import WorkflowSpec

if TYPE_CHECKING:
    from .registry import TaskTypeRegistry

__all__ = [
    "WorkflowCompiler",
    "default_compiler",
]


_SCRIPT_HEADER = '''\
"""Auto-generated molexp workflow plan.

The IR literal below is the source of truth — edit it, and the
matching ``WorkflowSpec.from_dict(WORKFLOW_IR)`` call below will
load your edits. The server reads the IR (not this script) when
running the workflow; the script exists so the IR is comfortable
to read and review in Python form.
"""
from molexp.workflow.spec import WorkflowSpec
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
        and then loads it via :meth:`WorkflowSpec.from_dict`. Re-parsing
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
            f"{_SCRIPT_HEADER}\n"
            f"WORKFLOW_IR = {literal}\n\n"
            f"spec = WorkflowSpec.from_dict(WORKFLOW_IR)\n"
        )

    def python_to_ir(self, script: str) -> JSONMapping:
        """Extract the IR literal from a Python script.

        Walks the module AST looking for a top-level
        ``WORKFLOW_IR = <dict literal>`` and decodes it via
        :func:`ast.literal_eval`. Anything else in the script (imports,
        comments, the trailing ``WorkflowSpec.from_dict`` call) is
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

    # ── IR ↔ WorkflowSpec ───────────────────────────────────────────────

    def ir_to_spec(
        self, ir: JSONMapping, *, registry: TaskTypeRegistry | None = None
    ) -> WorkflowSpec:
        """Build a :class:`WorkflowSpec` from JSON IR.

        Thin wrapper over :meth:`WorkflowSpec.from_dict`; exposed here
        so callers have one obvious entry point for *any* representation
        conversion.
        """
        return WorkflowSpec.from_dict(ir, registry=registry)

    def spec_to_ir(self, spec: WorkflowSpec) -> JSONMapping:
        """Serialize a :class:`WorkflowSpec` to JSON IR."""
        return spec.to_dict()

    # ── Spec → Python / Mermaid (composition) ───────────────────────────

    def spec_to_python(self, spec: WorkflowSpec) -> str:
        """Render a spec as a Python script (via the IR)."""
        return self.ir_to_python(self.spec_to_ir(spec))

    def spec_to_mermaid(self, spec: WorkflowSpec) -> str:
        """Render a spec as a Mermaid flowchart (via the IR)."""
        return self.ir_to_mermaid(self.spec_to_ir(spec))

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
