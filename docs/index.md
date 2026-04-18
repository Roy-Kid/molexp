# MolExp Documentation

MolExp's docs are organized into a small number of stable sections so readers can move from orientation to execution to internals without bouncing across many unrelated top-level folders.

## Documentation Structure

```
docs/
├── getting-started/   # concepts and orientation
├── tutorial/          # end-to-end hands-on walkthroughs
├── guide/             # focused user and operator guides
├── development/       # contributor-facing internals
└── spec/              # active design notes and implementation plans
```

## Start Here

<div class="grid cards" markdown>

-   :material-compass-outline: **Getting Started**

    ---

    Start with the conceptual model: workflow, workspace, profiles, and optional plugins.

    [:octicons-arrow-right-24: Open Overview](getting-started/overview.md)

-   :material-rocket-launch: **Tutorial**

    ---

    Follow one complete workflow from authoring through tracked execution.

    [:octicons-arrow-right-24: Open Quick Start](tutorial/quick-start.md)

-   :material-book-open-page-variant: **Guides**

    ---

    Use focused guides for runtime behavior, workspace persistence, server operations, and scheduler submission.

    [:octicons-arrow-right-24: Open Guide Index](guide/index.md)

-   :material-tools: **Development**

    ---

    Read the contributor-facing internals when you need compiler, IR, or protocol details.

    [:octicons-arrow-right-24: Open Development Index](development/index.md)

</div>

## Sections

- **Getting started**
  - [Overview](getting-started/overview.md)
- **Tutorial**
  - [Quick Start](tutorial/quick-start.md)
- **Guide**
  - [Guide index](guide/index.md)
  - [Run Profiles and Reproducible CLI Execution](guide/run-profiles.md)
  - [Workspace Architecture](guide/workspace-architecture.md)
  - [Workspace API](guide/workspace-api.md)
  - [Task and Actor](guide/task-and-actor.md)
  - [TaskContext](guide/task-context.md)
  - [Workflow Runtime](guide/workflow-runtime.md)
  - [Control Flow](guide/control-flow.md)
  - [Assets](guide/assets.md)
  - [Workflow Persistence](guide/workflow-persistence.md)
  - [Server Lifecycle](guide/server-lifecycle.md)
  - [Molq Plugin and Cluster Submission](guide/molq.md)
- **Development**
  - [Development index](development/index.md)
  - [Compiler](development/compiler.md)
  - [IR and Compiler Notes](development/ir-and-compiler.md)
  - [Task Protocols](development/task-protocols.md)
- **Spec**
  - [molcfg profiles](spec/molcfg-profiles.md)
  - [Unified pydantic-graph dispatch](spec/unified-pydantic-graph-dispatch.md)
  - [Fullscreen monitor](spec/fullscreen-monitor.md)

## Suggested Reading Order

If you are new to the project, start with [Overview](getting-started/overview.md), then do the [Quick Start](tutorial/quick-start.md), then read the [Run Profiles guide](guide/run-profiles.md). If you are implementing workflows, continue into the guide section. If you are changing compiler behavior or internal contracts, move into [Development](development/index.md). `spec/` is intentionally separate because it holds active design work rather than stable user documentation.
