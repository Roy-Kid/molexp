# Plugins

The plugin layer is how MolExp grows beyond the local core without forcing every installation to carry every optional dependency. The guiding rule is simple: `import molexp` should still work in a lightweight environment. Heavy integrations are loaded only when the user actually asks for the relevant capability.

## Why Plugins Exist

This boundary protects the core workflow and workspace model. Local authoring, local execution, and workspace inspection should not fail just because a machine does not have cluster tooling or agent-specific dependencies installed. Plugins make it possible to add those capabilities without turning the base package into a bundle of unrelated operational concerns.

## The Plugins in This Repository

A small registry of optional capabilities ships today. `submit_molq` is the scheduler bridge used when `molexp run` needs to submit work to Slurm, PBS, LSF, or another `molq`-backed scheduler. `gh` is a lazy GitHub client. `tensorboard` (installed via `molexp[tensorboard]`) parses tfevents into typed Python. In every case the core package stays unaware of those dependencies until the user reaches for the capability.

The **agent is not a plugin** — it is a first-class layer (`molexp.agent`) gated behind the `molexp[agent]` extra, the same way the rest of the core stays light without it. Beyond this registry, molexp also exposes two **third-party** extension channels — CLI subcommands (`molexp.cli_plugins`) and dynamically-imported UI bundles (`molexp.ui_plugins`) — so a downstream package can extend molexp without forking it. See [Writing a Plugin](../plugins.md) for those channels.

## Scheduler Transport Is Not a Second Runtime

The most important example is `submit_molq`, because it can easily look larger than it really is. The plugin does not define task semantics, replace `Workflow`, or invent a second persistence model. Its job is narrower. It translates CLI scheduling flags into a job submission, launches `python -m molexp.cli execute <run_dir>` on the target scheduler, and writes normalized executor metadata back onto the run record.

That design is why local and remote execution remain conceptually aligned. The workflow is still the same workflow. The workspace record is still the same workspace record. Only the transport layer changes.

## The Agent Layer Follows the Same Rule

The agent layer is not a plugin, but it honours the same boundary. It adds a public surface for goal-driven sessions and tool orchestration, yet it stores its durable state in the same workspace hierarchy. Sessions, approvals, tool calls, and observations remain inspectable through the same API and UI because the layer extends the persistent model instead of bypassing it — and `import molexp` stays light because none of pydantic-ai loads until you actually run an agent. See the [Agent concept](agent.md) for that layer in full.

If you need the operational detail behind scheduler submission, continue with [Molq Plugin](../guide/molq.md). If you need the server surface that exposes plugin-backed state to the UI, continue with [Server Lifecycle](../guide/server-lifecycle.md).
