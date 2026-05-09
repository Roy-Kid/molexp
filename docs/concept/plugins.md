# Plugins

The plugin layer is how MolExp grows beyond the local core without forcing every installation to carry every optional dependency. The guiding rule is simple: `import molexp` should still work in a lightweight environment. Heavy integrations are loaded only when the user actually asks for the relevant capability.

## Why Plugins Exist

This boundary protects the core workflow and workspace model. Local authoring, local execution, and workspace inspection should not fail just because a machine does not have cluster tooling or agent-specific dependencies installed. Plugins make it possible to add those capabilities without turning the base package into a bundle of unrelated operational concerns.

## The Plugins in This Repository

Two optional integrations ship with the repository today. `submit_molq` is the scheduler bridge used when `molexp run` needs to submit work to Slurm, PBS, LSF, or another `molq`-backed scheduler. `agent_pydanticai` is the agent integration layer for goal-driven execution built on top of PydanticAI. In both cases the core package can remain unaware of those dependencies until the user reaches for them.

## Scheduler Transport Is Not a Second Runtime

The most important example is `submit_molq`, because it can easily look larger than it really is. The plugin does not define task semantics, replace `Workflow`, or invent a second persistence model. Its job is narrower. It translates CLI scheduling flags into a job submission, launches `python -m molexp.cli execute <run_dir>` on the target scheduler, and writes normalized executor metadata back onto the run record.

That design is why local and remote execution remain conceptually aligned. The workflow is still the same workflow. The workspace record is still the same workspace record. Only the transport layer changes.

## Agent Integration Follows the Same Rule

The agent plugin follows the same philosophy. It adds a public surface for goal-driven sessions and tool orchestration, but it still stores its durable state in the same workspace hierarchy. Sessions, approvals, tool calls, and observations remain inspectable through the same API and UI because the plugin extends the persistent model instead of bypassing it.

If you need the operational detail behind scheduler submission, continue with [Molq Plugin](../guide/molq.md). If you need the server surface that exposes plugin-backed state to the UI, continue with [Server Lifecycle](../guide/server-lifecycle.md).
