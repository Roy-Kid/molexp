---
slug: workflow-refactor-01-codec-fold
criteria:
  - id: ac-001
    summary: The "WorkflowCompiler" name is freed from the public workflow API
    type: code
    pass_when: |
      `from molexp.workflow import WorkflowCompiler` raises ImportError and
      "WorkflowCompiler" is absent from molexp.workflow.__all__; a grep gate
      (`grep -rn "WorkflowCompiler" src/molexp/workflow/`) returns no class
      definition or export. Asserted in tests/test_workflow/test_codec.py.
    status: pending
  - id: ac-002
    summary: WorkflowCodec + default_codec are the public codec surface
    type: code
    pass_when: |
      `from molexp.workflow import WorkflowCodec, default_codec` succeeds;
      WorkflowCodec is the class formerly named WorkflowCompiler (same
      stateless converter contract, instance methods intact). Asserted in
      tests/test_workflow/test_codec.py.
    status: pending
  - id: ac-003
    summary: IR / Python / Mermaid output is byte-identical to pre-refactor
    type: code
    pass_when: |
      For every existing workflow fixture, default_codec.spec_to_ir(spec),
      ir_to_python(ir), and ir_to_mermaid(ir) produce output byte-identical to
      a captured main-branch golden; spec_to_ir(ir_to_spec(ir)) == ir round-trips
      for the slugged data-DAG fixtures. Asserted in
      tests/test_workflow/test_codec.py and passes under
      pytest tests/test_workflow/.
    status: pending
  - id: ac-004
    summary: WorkflowCodec is the single owner of IR conversion
    type: code
    pass_when: |
      The to_dict/from_dict logic lives in WorkflowCodec.spec_to_ir/ir_to_spec;
      Workflow.to_dict/from_dict are one-line delegators to default_codec
      (verified by source inspection / a unit test asserting Workflow.to_dict(s)
      == default_codec.spec_to_ir(s)); server/routes/execution.py calls
      default_codec.ir_to_spec(...) rather than Workflow.from_dict(...).
    status: pending
  - id: ac-005
    summary: No execution-path regression; full gate is green
    type: code
    pass_when: |
      `ruff format --check src/ tests/ && ruff check src/ tests/ && ty check src/
      && pytest tests/` all pass with coverage on workflow/codec.py at least
      equal to the prior serializer.py coverage; the workflow runtime test suite
      (test_runtime / test_spec) is unchanged and green.
    status: pending
---

# Acceptance — workflow-refactor-01-codec-fold

Behavior-preserving rename + codec consolidation. "Done" means the
`WorkflowCompiler` name is free for spec 02 (ac-001), the codec is renamed and is
the single IR owner (ac-002, ac-004), every representation surface is byte-identical
to `main` (ac-003), and the full quality gate is green with no execution-path
change (ac-005).
