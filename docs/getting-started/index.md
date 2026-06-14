# Getting Started

This section is the practical path through MolExp. It is organized around the order most people encounter the system in real work: first you want to run something, then you want to understand how that run is tracked, and only after that do you usually care about CLI discovery, profiles, and repeatable execution variants.

## The Short Path

Start with [Quick Start](quick-start.md) if you want one complete example that goes from a workflow definition to a tracked run with the least ceremony possible. That page is intentionally narrow. Its job is to show the shape of the system, not to explain every option.

If you would rather click than script, [Start from the UI](start-from-ui.md) walks the same lifecycle through the browser: serve a workspace, create a project and an experiment through dialogs, launch a run, and manage it from the run page. It also states plainly which management operations are not in the UI yet.

Then read [Your First Workflow](first-workflow.md) to understand what a workflow is before a workspace is attached to it. Continue with [Track a Run](tracked-runs.md) when you want the persistent hierarchy to make sense, and then read [CLI and Profiles](cli-and-profiles.md) when you are ready to run the same script from `molexp run` and keep execution variants in `molcfg.yaml`.

## When to Leave This Section

Once you can read a script that calls `wf.compile()` and declares the result on an experiment with `ws.project(...).experiment(...).run(compiled, params=...)` — the fluent chain `molexp run` discovers — you are ready to leave onboarding. At that point the [Concepts](../concept/index.md) section will help firm up the model, and the [Guide](../guide/index.md) section will be the better place for detailed topics.
