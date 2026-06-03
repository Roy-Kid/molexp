---
slug: flowgram-workflow-canvas-03-read-canvas
criteria:
  - id: ac-001
    summary: Zero @xyflow/elkjs/dagre matches anywhere under ui/src/
    type: code
    pass_when: |
      `rg -n "@xyflow|elkjs|dagre" ui/src/` returns no matches
      (exit code 1, empty output).
    status: pending
  - id: ac-002
    summary: ElkEdge/elkLayout/dagreLayout modules deleted
    type: code
    pass_when: |
      None of ui/src/app/renderers/ElkEdge.tsx, elkLayout.ts,
      dagreLayout.ts exist on disk.
    status: pending
  - id: ac-003
    summary: package.json swaps xyflow/elk/dagre for flowgram + reflect-metadata
    type: code
    pass_when: |
      ui/package.json dependencies contain
      "@flowgram.ai/free-layout-editor": "1.0.11" and "reflect-metadata",
      and contain none of @xyflow/react, elkjs, @dagrejs/dagre.
    status: pending
  - id: ac-004
    summary: task_graph_ir.ts is the sole IR type, reshaped to the 01 shape
    type: code
    pass_when: |
      ui/src/types/task_graph_ir.ts exports TaskGraphJson/TaskNodeJson/EdgeJson
      with task_configs + typed (kind) links + node position; and no second
      {nodes,edges} WorkflowGraph IR interface remains anywhere under ui/src/
      (the WorkflowGraph/WorkflowNodeMetadata/WorkflowGraphEdge shapes are gone
      from app/types.ts).
    status: pending
  - id: ac-005
    summary: buildFlowgramDocument builds a well-formed doc from representative new IR
    type: code
    pass_when: |
      WorkflowGraph.test.ts passes: for a multi-node IR with typed links and
      explicit position, buildFlowgramDocument emits one document node per
      task_config (with position+data) and one edge per valid link.
    status: pending
  - id: ac-006
    summary: builder drops invalid links and survives cyclic IR
    type: code
    pass_when: |
      WorkflowGraph.test.ts passes: links referencing unknown node ids are
      dropped from the document; a cyclic IR produces a document where every
      node still has a numeric position and the builder does not throw.
    status: pending
  - id: ac-007
    summary: flowgram canvas renders and node click opens TaskViewer via inspectedTask
    type: ui_runtime
    evaluator_hint: mol:web
    pass_when: |
      In the running UI, selecting a workflow file/run renders the flowgram
      read-only canvas with task nodes and links; clicking a node sets
      inspectedTask and opens TaskViewer in the right panel; plugin panels
      still load.
    status: pending
  - id: ac-008
    summary: npm typecheck + test + build green with decorator metadata
    type: runtime
    pass_when: |
      `cd ui && npm run typecheck && npm run test && npm run build` all exit 0
      with reflect-metadata imported and emitDecoratorMetadata enabled.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003** — the hard removal bar: a single `rg` over `ui/src/` plus disk checks plus the dependency swap. These three together encode "complete refactor, no backward-compat layer."
- **ac-004** — single-IR-type invariant. Verifies both that `task_graph_ir.ts` was reshaped to 01's `task_configs`/`links`/`kind`/`position` and that the legacy `{nodes,edges}` `WorkflowGraph` family was deleted from `app/types.ts` (no hedge IR).
- **ac-005 / ac-006** — the retargeted `WorkflowGraph.test.ts` unit suite against flowgram-document output (well-formed doc, invalid-link drop, cyclic survival).
- **ac-007** — runtime evaluator (mol:web): the canvas actually renders and the `inspectedTask` → `TaskViewer` click path still works, with plugins unaffected.
- **ac-008** — full green build proving the decorator-metadata toolchain wiring is correct.
