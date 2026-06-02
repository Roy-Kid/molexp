---
slug: workflow-refactor-02-merge-build-compile
criteria:
  - id: ac-001
    summary: compile() emits a CompiledWorkflow carrying graph + snapshots + version + binding
    type: code
    pass_when: |
      WorkflowCompiler(...).compile(experiment=exp, registry=r) returns a
      CompiledWorkflow whose .snapshots has exactly one TaskSnapshot per
      registered task, .version is a populated WorkflowVersion, .graph is
      non-None, and .binding references exp. Asserted in
      tests/test_workflow/test_compiler.py.
    status: pending
  - id: ac-002
    summary: build + compile merged; WorkflowBuilder / Workflow / WorkflowGraphCompiler gone from the public API
    type: code
    pass_when: |
      `from molexp.workflow import WorkflowBuilder` and `import Workflow` both
      raise ImportError; WorkflowCompiler + CompiledWorkflow are exported; a grep
      gate finds no public WorkflowGraphCompiler export and no discarded-compile
      intermediate in compile(). Asserted in tests/test_workflow/test_compiler.py.
    status: pending
  - id: ac-003
    summary: binding via explicit WorkflowBindingRegistry, no process-global
    type: code
    pass_when: |
      compile(experiment=exp, registry=r) makes r.for_experiment(exp) is the
      compiled artifact; symbols Workflow._bindings_registry and _reset_registry
      do not exist (grep gate); server/cli/entry callers use the registry.
      Asserted in tests/test_workflow/test_binding.py.
    status: pending
  - id: ac-004
    summary: codec folded onto CompiledWorkflow; IR round-trip holds
    type: code
    pass_when: |
      CompiledWorkflow.to_ir() then CompiledWorkflow.from_ir(...) round-trips to
      an equal artifact for slugged data-DAG fixtures, byte-identical to the 01
      WorkflowCodec output. Asserted in tests/test_workflow/test_compiler.py.
    status: pending
  - id: ac-005
    summary: execution outputs unchanged via runtime.execute(compiled)
    type: code
    pass_when: |
      For every existing workflow fixture (chain/parallel/branch/loop/reduce),
      GraphWorkflowRuntime.execute(compiled) yields WorkflowResult.outputs equal
      to the pre-refactor Workflow.execute() golden. Asserted in
      tests/test_workflow/test_runtime.py.
    status: pending
  - id: ac-006
    summary: full quality gate is green
    type: code
    pass_when: |
      `ruff format --check src/ tests/ && ruff check src/ tests/ && ty check src/
      && pytest tests/` all pass.
    status: pending
---

# Acceptance — workflow-refactor-02-merge-build-compile

"Done" means one `WorkflowCompiler.compile()` emits a `CompiledWorkflow` holding
graph + snapshots + version + binding (ac-001), the old build/compile/spec
classes are gone from the public API (ac-002), binding is explicit-registry-based
with no global (ac-003), the codec rides on `CompiledWorkflow` with an intact IR
round-trip (ac-004), execution output is unchanged (ac-005), and the gate is
green (ac-006).
