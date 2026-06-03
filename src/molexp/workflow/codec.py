"""Workflow codec — convert between IR, Python script, Mermaid, YAML, and Spec.

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

:class:`WorkflowCodec` provides pairwise converters between these
surfaces and round-trip guarantees where they apply::

    ir = codec.python_to_ir(codec.ir_to_python(ir))  # exact
    ir = codec.spec_to_ir(codec.ir_to_spec(ir))  # exact (slugged tasks)
    text = codec.ir_to_mermaid(ir)  # one-way

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
from ._graph_decl import LoopDecl, ParallelDecl, TaskRegistration
from ._helpers import _ir_object_list, _require_str
from .contract import WorkflowContract
from .protocols import Streamable

if TYPE_CHECKING:
    from .compiled import CompiledWorkflow
    from .registry import TaskTypeRegistry

__all__ = [
    "WorkflowCodec",
    "default_codec",
]


_SCRIPT_HEADER = '''\
"""Auto-generated molexp workflow plan.

The IR literal below is the source of truth — edit it, and the
matching ``CompiledWorkflow.from_ir(WORKFLOW_IR)`` call below will
load your edits. The server reads the IR (not this script) when
running the workflow; the script exists so the IR is comfortable
to read and review in Python form.
"""
from molexp.workflow import CompiledWorkflow
'''


class WorkflowCodec:
    """Stateless converter between workflow representations.

    Methods are instance methods (rather than classmethods) so callers
    can subclass and override individual conversions — e.g. swap the
    Mermaid renderer for a Graphviz one — without touching the rest of
    the API. The default instance is also exposed at module level as
    :data:`default_codec` for callers that don't need customization.

    This codec is the single owner of IR conversion: ``spec_to_ir`` /
    ``ir_to_spec`` hold the authoritative ``CompiledWorkflow`` ⇄ IR bodies,
    and ``CompiledWorkflow.to_ir`` / ``from_ir`` are thin delegators to
    :data:`default_codec`.
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
        return f"{_SCRIPT_HEADER}\nWORKFLOW_IR = {literal}\n\nspec = CompiledWorkflow.from_ir(WORKFLOW_IR)\n"

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

    # ── IR ↔ CompiledWorkflow ────────────────────────────────────────

    def ir_to_spec(
        self, ir: JSONMapping, *, registry: TaskTypeRegistry | None = None
    ) -> CompiledWorkflow:
        """Build a :class:`CompiledWorkflow` from JSON IR.

        This is the authoritative IR → :class:`CompiledWorkflow` body;
        :meth:`CompiledWorkflow.from_ir` delegates here.
        """
        if registry is None:
            from .registry import default_registry

            registry = default_registry

        links_raw = _ir_object_list(ir.get("links"))
        task_configs_raw = _ir_object_list(ir.get("task_configs"))

        # Split the typed edge set by ``kind``. A missing ``kind`` is read as
        # ``data`` so hand-authored IR stays loadable; the JSON-Schema marks
        # ``kind`` required, so the strict contract is enforced at validation.
        deps_by_target: dict[str, list[str]] = {}
        control_edges: list[tuple[str, str]] = []
        branch_edges: list[tuple[str, str, str]] = []
        for link in links_raw:
            source = _require_str(link, "source")
            target = _require_str(link, "target")
            kind_raw = link.get("kind")
            kind = kind_raw if isinstance(kind_raw, str) else "data"
            if kind == "control":
                control_edges.append((source, target))
            elif kind == "branch":
                cond_raw = link.get("condition")
                condition = cond_raw if isinstance(cond_raw, str) else ""
                branch_edges.append((source, condition, target))
            else:  # data (and any unrecognized kind) is a dependency edge
                deps_by_target.setdefault(target, []).append(source)

        tasks: list[TaskRegistration] = []
        for tc in task_configs_raw:
            slug = _require_str(tc, "task_type")
            task_id = _require_str(tc, "task_id")
            cfg_raw = tc.get("config")
            config: dict[str, JSONValue] = dict(cfg_raw) if isinstance(cfg_raw, dict) else {}
            factory = registry.get(slug)
            instance = factory(config)
            tasks.append(
                TaskRegistration(
                    name=task_id,
                    fn_or_class=instance,
                    depends_on=deps_by_target.get(task_id, []),
                    is_actor=isinstance(instance, Streamable),
                    task_type=slug,
                    config=config,
                    position=_read_position(tc.get("position")),
                )
            )

        known = {t.name for t in tasks}
        for link in links_raw:
            for endpoint in (_require_str(link, "source"), _require_str(link, "target")):
                if endpoint not in known:
                    raise ValueError(
                        f"Link references unknown task_id {endpoint!r}; known: {sorted(known)}"
                    )

        entries_raw = ir.get("entries")
        entries = (
            tuple(s for s in entries_raw if isinstance(s, str))
            if isinstance(entries_raw, list)
            else ()
        )
        loops = tuple(
            LoopDecl(
                body=_str_tuple(loop.get("body")),
                until=_require_str(loop, "until"),
                max_iters=_require_int(loop, "max_iters"),
                on_exit=_require_str(loop, "on_exit"),
            )
            for loop in _ir_object_list(ir.get("loops"))
        )
        parallels = tuple(
            ParallelDecl(
                map_over=_require_str(p, "map_over"),
                body=_require_str(p, "body"),
                join=_require_str(p, "join"),
                max_concurrency=_require_int(p, "max_concurrency"),
            )
            for p in _ir_object_list(ir.get("parallels"))
        )

        name_raw = ir.get("name")
        name = name_raw if isinstance(name_raw, str) else ""
        from .compiler import compile_registrations

        return compile_registrations(
            name=name,
            version_label="0",
            tasks=tasks,
            entries=entries,
            control_edges=tuple(control_edges),
            branch_edges=tuple(branch_edges),
            loops=loops,
            parallels=parallels,
        )

    def spec_to_ir(self, spec: CompiledWorkflow) -> JSONMapping:
        """Serialize a :class:`Workflow` to the JSON IR shape (see ``schema/workflow.json``).

        This is the authoritative :class:`Workflow` → IR body;
        :meth:`Workflow.to_dict` delegates here.
        """
        unslugged = [t.name for t in spec._tasks if t.task_type is None]
        if unslugged:
            raise ValueError(
                "Cannot serialize workflow to IR: the following tasks have no "
                f"task_type slug: {unslugged}. Use `WorkflowBuilder.add(..., task_type=...)` "
                "or build the spec from IR via CompiledWorkflow.from_ir()."
            )
        task_configs: list[JSONValue] = []
        for t in spec._tasks:
            tc: dict[str, JSONValue] = {
                "task_id": t.name,
                "task_type": t.task_type,
                "config": dict(t.config) if t.config else {},
                "status": "pending",
            }
            position = getattr(t, "position", None)
            if position is not None:
                x, y = position
                tc["position"] = {"x": x, "y": y}
            task_configs.append(tc)
        # Typed edges: data deps + control + branch. Loops / parallels carry
        # extra structure (max_iters, map_over, …) a flat link can't hold, so
        # they ride the structured top-level arrays below — that is what makes
        # ``ir_to_spec`` a lossless inverse.
        links: list[JSONValue] = []
        for t in spec._tasks:
            for dep in t.depends_on:
                links.append(_link(dep, t.name, "data"))
        for src, tgt in spec._control_edges:
            links.append(_link(src, tgt, "control"))
        for src, label, tgt in spec._branch_edges:
            links.append(_link(src, tgt, "branch", condition=label))
        loops: list[JSONValue] = [
            {
                "body": list(loop.body),
                "until": loop.until,
                "max_iters": loop.max_iters,
                "on_exit": loop.on_exit,
            }
            for loop in spec._loops
        ]
        parallels: list[JSONValue] = [
            {
                "map_over": p.map_over,
                "body": p.body,
                "join": p.join,
                "max_concurrency": p.max_concurrency,
            }
            for p in spec._parallels
        ]
        metadata: dict[str, JSONValue] = {
            "label": None,
            "description": None,
            "tags": [],
            "custom": {},
        }
        return {
            "workflow_id": f"workflow_{spec.workflow_id[:8]}",
            "name": spec.name,
            "task_configs": task_configs,
            "links": links,
            "entries": list(spec._entries),
            "loops": loops,
            "parallels": parallels,
            "metadata": metadata,
        }

    # ── Spec → Python / Mermaid (composition) ───────────────────────────

    def spec_to_python(self, spec: CompiledWorkflow) -> str:
        """Render a spec as a Python script (via the IR)."""
        return self.ir_to_python(self.spec_to_ir(spec))

    def spec_to_mermaid(self, spec: CompiledWorkflow) -> str:
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

    def spec_to_yaml(self, spec: CompiledWorkflow) -> str:
        """Render a spec as YAML text (via the JSON IR)."""
        return self.ir_to_yaml(self.spec_to_ir(spec))

    def yaml_to_spec(
        self, text: str, *, registry: TaskTypeRegistry | None = None
    ) -> CompiledWorkflow:
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


default_codec = WorkflowCodec()


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


def _link(source: str, target: str, kind: str, *, condition: str | None = None) -> JSONValue:
    """Build one typed wire-IR link object."""
    link: dict[str, JSONValue] = {
        "source": source,
        "target": target,
        "mapping": {},
        "status": "pending",
        "kind": kind,
    }
    if condition is not None:
        link["condition"] = condition
    return link


def _str_tuple(raw: JSONValue | None) -> tuple[str, ...]:
    """Narrow an IR field expected to hold a list of strings into a tuple."""
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw)


def _require_int(obj: dict[str, JSONValue], key: str) -> int:
    """Read an integer-valued IR field, rejecting missing / non-numeric values."""
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"IR field {key!r} must be a number; got {value!r}.")
    return int(value)


def _read_position(raw: JSONValue | None) -> tuple[float, float] | None:
    """Decode a ``{"x": .., "y": ..}`` IR position object into an ``(x, y)`` tuple."""
    if not isinstance(raw, dict):
        return None
    x = raw.get("x")
    y = raw.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return (float(x), float(y))
    return None


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
