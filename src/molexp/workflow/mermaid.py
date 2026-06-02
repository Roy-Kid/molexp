"""Mermaid renderer for :class:`~molexp.workflow.ir.WorkflowGraphIR`.

One-way rendering of the full compiled-graph IR to a Mermaid ``flowchart``
block, suitable for docs, PR review, and read-only UI surfaces. Unlike the
DAG-only renderer in :mod:`.serializer` (which draws ``task_configs`` +
``links``), this one understands the complete topology: entries, dependency
edges, explicit control edges, label-routed branches, loops, and parallel
fan-outs.

Edge vocabulary in the rendered diagram:

- ``A --> B``           — B depends on A, or an unconditional control edge.
- ``A -->|label| B``    — a branch route: ``A`` emits ``label`` to reach ``B``.
- ``A -.->|continue| B``/``A -->|exit| B`` — a loop's back-edge and its exit.
- ``A -->|fan-out xN| B`` / ``B -->|join| C`` — a parallel map / join.

The function is pure and total — it never raises on a well-formed IR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ir import WorkflowGraphIR

__all__ = ["render_workflow_mermaid"]

_START_NODE = "__start"


def render_workflow_mermaid(ir: WorkflowGraphIR, *, direction: str = "LR") -> str:
    """Render *ir* as a ``flowchart`` Mermaid block.

    Args:
        ir: The compiled-graph IR to draw.
        direction: Mermaid flow direction (``LR`` / ``TD`` / ``RL`` / ``BT``).
    """
    lines = [f"flowchart {direction}"]

    # ── Entry marker ──────────────────────────────────────────────────────
    if ir.entries:
        lines.append(f"  {_START_NODE}((start))")
        for entry in ir.entries:
            lines.append(f"  {_START_NODE} --> {_node_id(entry)}")

    # ── Nodes ─────────────────────────────────────────────────────────────
    for task in ir.tasks:
        nid = _node_id(task.name)
        label = _node_label(task.name, task.task_type)
        if task.is_actor:
            # Stadium shape marks a streaming actor.
            lines.append(f'  {nid}(["{label}"])')
        else:
            lines.append(f'  {nid}["{label}"]')

    # ── Dependency + unconditional control edges (dedup) ──────────────────
    # Seed with branch-routed pairs: the labeled branch edge below is more
    # informative, so a plain dependency/control edge on the same pair is
    # suppressed rather than drawn twice.
    seen: set[tuple[str, str]] = {(src, tgt) for src, _label, tgt in ir.branch_edges}
    for task in ir.tasks:
        for dep in task.depends_on:
            edge = (dep, task.name)
            if edge not in seen:
                seen.add(edge)
                lines.append(f"  {_node_id(dep)} --> {_node_id(task.name)}")
    for src, tgt in ir.control_edges:
        edge = (src, tgt)
        if edge not in seen:
            seen.add(edge)
            lines.append(f"  {_node_id(src)} --> {_node_id(tgt)}")

    # ── Branch routes (labeled) ───────────────────────────────────────────
    for src, label, tgt in ir.branch_edges:
        lines.append(f"  {_node_id(src)} -->|{_edge_label(label)}| {_node_id(tgt)}")

    # ── Loops ─────────────────────────────────────────────────────────────
    for loop in ir.loops:
        if loop.body:
            lines.append(f"  {_node_id(loop.until)} -.->|continue| {_node_id(loop.body[0])}")
        lines.append(f"  {_node_id(loop.until)} -->|exit| {_node_id(loop.on_exit)}")

    # ── Parallel fan-outs ─────────────────────────────────────────────────
    for par in ir.parallels:
        fan = f"fan-out x{par.max_concurrency}" if par.max_concurrency > 1 else "fan-out"
        lines.append(f"  {_node_id(par.map_over)} -->|{fan}| {_node_id(par.body)}")
        lines.append(f"  {_node_id(par.body)} -->|join| {_node_id(par.join)}")

    return "\n".join(lines) + "\n"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _node_id(name: str) -> str:
    """Coerce a task name into a Mermaid-safe node identifier."""
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    return f"n_{safe}" if safe else "n_unnamed"


def _node_label(name: str, task_type: str | None) -> str:
    """Build a node label: name, plus an italic task_type subtitle if present."""
    base = _escape(name)
    if task_type:
        return f"{base}<br/><i>{_escape(task_type)}</i>"
    return base


def _edge_label(label: str) -> str:
    """Sanitize a branch label for use inside a Mermaid ``|...|`` edge label."""
    # ``|`` would close the edge-label delimiter; ``"`` is unsafe inside it.
    return _escape(label).replace("|", "/")


def _escape(text: str) -> str:
    """Replace characters that break Mermaid label literals."""
    return str(text).replace('"', "'")
