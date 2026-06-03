---
slug: flowgram-workflow-canvas-04-edit-writeback
criteria:
  - id: ac-001
    summary: Reverse serializer round-trips with the 03 forward builder (identity)
    type: code
    pass_when: |
      Test in ui/src/app/state/api.reverse-serialize.test.ts passes: on a
      representative TaskGraphJson fixture, buildFlowgramDocument∘reverse and
      reverse∘buildFlowgramDocument are field-equal (node id/type/config, edge
      from/to/kind, targets, position all preserved).
    status: pending
  - id: ac-002
    summary: Reverse serializer returns the single task_graph_ir.ts TaskGraphJson type
    type: code
    pass_when: |
      flowgramDocToTaskGraphJson's return type is TaskGraphJson imported from
      ui/src/types/task_graph_ir.ts (no second IR interface introduced); the
      reverse serializer and toTaskGraphJson both consume the shared mapping
      helpers in workflow-utils.ts. tsc passes with no duplicate IR type.
    status: pending
  - id: ac-003
    summary: Shared edge/targets mapping helpers reused, not duplicated
    type: code
    pass_when: |
      toTaskGraphJson is refactored to call the extracted edge->EdgeJson and
      isOutput->targets helpers, and the reverse serializer calls the same
      helpers; no copy of the edge/targets construction logic exists in api.ts.
    status: pending
  - id: ac-004
    summary: Canvas is editable while preserving 03 inspectedTask click path
    type: ui_runtime
    evaluator_hint: mol:web
    pass_when: |
      In the rendered flowgram canvas, a node can be dragged and an edge created;
      clicking a node still calls inspectTask and pins it to the right TaskViewer
      panel (03 behavior intact).
    status: pending
  - id: ac-005
    summary: Save action writes back via the generated client, not raw fetch
    type: ui_runtime
    evaluator_hint: mol:web
    pass_when: |
      Clicking the shadcn/ui save button calls the 02-regenerated service method
      in ui/src/api/generated/services/ with the reverse-serialized TaskGraphJson
      as payload; global fetch is not invoked by the save path (mocked service is
      called, mocked fetch is not).
    status: pending
  - id: ac-006
    summary: Edit -> write-back -> reload renders identically
    type: ui_runtime
    evaluator_hint: mol:web
    pass_when: |
      After an edit is saved and the canvas reloads from backend state, the
      rendered graph (nodes, edges, positions, targets) is semantically identical
      to the pre-reload edited state.
    status: pending
  - id: ac-007
    summary: Save toolbar built with shadcn/ui
    type: ui_runtime
    evaluator_hint: mol:web
    pass_when: |
      The save toolbar in FlowgramCanvasToolbar.tsx uses shadcn/ui components
      (e.g. Button from @/components/ui/button); any deviation is documented in
      the PR description.
    status: pending
  - id: ac-008
    summary: Full UI check + test suite passes
    type: runtime
    pass_when: |
      The UI lint/typecheck and test suite (npm run typecheck + npm test, or repo
      equivalent) pass with the new and modified files.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003 (code):** the reverse serializer is the verifiable inverse of 03's `buildFlowgramDocument`, reuses the single `task_graph_ir.ts` type set, and shares mapping helpers with `toTaskGraphJson` rather than duplicating IR construction.
- **ac-004 / ac-006 (ui_runtime, hint `mol:web`):** editing works and the 03 inspect path survives; the round-trip through the backend renders identically.
- **ac-005 (ui_runtime, hint `mol:web`):** the binding invariant — save goes through the generated client, never a hand-rolled fetch.
- **ac-007 (ui_runtime, hint `mol:web`):** new chrome uses shadcn/ui.
- **ac-008 (runtime):** the full check + test suite gate.
