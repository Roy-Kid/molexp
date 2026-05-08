# Workflow Layer Architecture

`molexp.workflow` is the only workflow abstraction in Molexp.

Every graph-like workflow must be represented through this layer:

- planning workflow
- plan revision workflow
- executable scientific workflow
- experiment workflow
- run materialization workflow
- repair workflow
- dry-run workflow

`molexp.workflow` may internally use `pydantic-graph`, but that dependency is
private. Code outside `molexp.workflow` must not import `pydantic_graph` or
`molexp.workflow._pydantic_graph`.

## Responsibilities

`molexp.workflow` owns:

- workflow definition
- workflow node definition
- workflow edge definition
- dependency graph
- control edges
- stable node identity
- node patching
- subgraph validation
- graph serialization
- graph rendering
- backend-independent compile and dry-run hooks

It does not own scheduler dispatch, job monitoring, workspace run directories,
or backend-specific logging.

## Editable Nodes

Every workflow node must have:

- stable `node_id`
- human-readable title
- node kind
- input schema
- output schema
- status
- provenance
- dependencies
- editable fields
- validation rules

The workflow layer must expose operations equivalent to:

- `get_node`
- `patch_node`
- `replace_node`
- `rewrite_node`
- `remove_node`
- `insert_node`
- `mark_downstream_stale`
- `validate_subgraph`
- `render_subgraph_preview`

Exact method names may evolve, but the capabilities are required.

## Public Boundary

Allowed outside `molexp.workflow`:

```python
from molexp.workflow import WorkflowSpec, WorkflowNode, WorkflowEdge
```

Forbidden outside `molexp.workflow`:

```python
from pydantic_graph import Graph, BaseNode
import pydantic_graph
```

The import boundary is enforced by tests.
