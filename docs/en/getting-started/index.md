# Getting Started

This section is the practical path. It is organized around the order most people encounter MolExp in real work: run something first, understand how it's tracked, then add CLI execution and profiles.

## Your path

1. **[Quick Start](quick-start.md)** — One script, two tasks, one tracked run. The fastest way to see the whole system work.
2. **[Your First Workflow](first-workflow.md)** — Tasks, dependencies, compilation, sync/async mixing. Understand the workflow before attaching a workspace.
3. **[Track a Run](tracked-runs.md)** — The persistent hierarchy (Workspace → Project → Experiment → Run). Sweeps, resume, rerun, and CLI registration.
4. **[CLI and Profiles](cli-and-profiles.md)** — Replace `asyncio.run()` with `molexp run`. Add `molcfg.yaml` for execution variants.
5. **[Start from the UI](start-from-ui.md)** — Create projects, experiments, and runs from the browser. No Python required.

## When you're ready to go deeper

Once you can read a script that calls `wf.compile()` and `run.execute(wf)`, the [Concepts](../concept/index.md) section will firm up the mental model, and the [Guide](../guide/index.md) section covers detailed topics.
