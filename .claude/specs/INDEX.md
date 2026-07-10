# Specs Index
One line per **live** spec. `/mol:spec` adds entries; `/mol:impl` ticks the spec's tasks and prunes the entry (with the spec file) on completion.

## agent-code-loop chain (implement in order)

- [agent-code-loop-02-code-tools](agent-code-loop-02-code-tools.md) — write_file + execute_python 始终挂载 [approved]
- [agent-code-loop-03-mcp-wire](agent-code-loop-03-mcp-wire.md) — stream_agentic toolsets + McpStore 接线 [approved]
- [agent-code-loop-04-molmcp-scaffold](agent-code-loop-04-molmcp-scaffold.md) — molmcp MolexpProvider 脚手架工具 [approved]
- [agent-code-loop-05-behavior](agent-code-loop-05-behavior.md) — 咨询→写码→执行 行为规范 [approved]

_Closed: agent-code-loop-01-golden (Python 金路径 example + pytest)._
_Closed product-gap remediation (2026-07) and agent-record-export 01–08._
Closed without external work: `execution-semantics` (implemented in-tree: `ctx.workdir`, `Experiment.run`/`sweep`, `execute_run`/`RunSet.execute`, materialization store).
Closed as blocked: `pure-task-context-03-build-flow-rewrite` — target `/Users/roykid/work/molcrafts/polymer_electrolyte/build_flow.py` is absent; workdir contract is `ctx.workdir` (not `ctx.inputs`) per current engine.
