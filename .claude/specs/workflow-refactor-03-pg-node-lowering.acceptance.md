---
slug: workflow-refactor-03-pg-node-lowering
criteria:
  - id: ac-001
    summary: the hand-rolled scheduler is deleted
    type: code
    pass_when: |
      A grep gate over src/molexp/workflow/_pydantic_graph/ finds no WorkflowStep
      class, no _dispatch method, and no level_index loop; the module imports and
      uses pydantic_graph Fork/Join/Decision/GraphBuilder. Asserted in
      tests/test_workflow/test_pg_lowering.py.
    status: pending
  - id: ac-002
    summary: compile emits a genuine pydantic_graph.Graph
    type: code
    pass_when: |
      CompiledWorkflow.graph is an instance of pydantic_graph.Graph with one node
      per task and edges matching depends_on / entries matching wf.entry.
      Asserted in tests/test_workflow/test_pg_lowering.py.
    status: pending
  - id: ac-003
    summary: parallel/branch/reduce lower to pydantic-graph primitives with identical semantics
    type: code
    pass_when: |
      wf.parallel produces Fork/Join (ordered per-element results + bounded
      concurrency preserved); wf.branch produces Decision (correct route);
      wf.reduce produces a reduce_* reducer (correct aggregation). Asserted in
      tests/test_workflow/test_pg_lowering.py.
    status: pending
  - id: ac-004
    summary: outputs byte-identical to the pre-03 executor
    type: code
    pass_when: |
      For every fixture (chain/parallel/branch/loop/reduce), WorkflowResult.outputs
      equals a captured pre-03 golden. Asserted in tests/test_workflow/test_runtime.py.
    status: pending
  - id: ac-005
    summary: stall and cycle still raise
    type: code
    pass_when: |
      A cyclic graph and a stalled (unsatisfiable-dependency) graph each raise a
      workflow error (surfaced by pydantic-graph, mapped to the existing error
      type). Asserted in tests/test_workflow/test_pg_lowering.py.
    status: pending
  - id: ac-006
    summary: full quality gate is green
    type: code
    pass_when: |
      `ruff format --check src/ tests/ && ruff check src/ tests/ && ty check src/
      && pytest tests/` all pass.
    status: pending
---

# Acceptance — workflow-refactor-03-pg-node-lowering

"Done" means the reinvented scheduler is gone (ac-001), `CompiledWorkflow.graph`
is a real pydantic-graph `Graph` (ac-002), control flow rides pydantic-graph
primitives with identical semantics (ac-003), every fixture's outputs are
byte-identical (ac-004), stall/cycle still error (ac-005), and the gate is green
(ac-006). Internal swap only — no public-surface change.
