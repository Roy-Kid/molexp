# MolExp

MolExp is a workflow system for research execution, persistent experiment records, and managed reusable assets. It is built around a simple separation of concerns. The workflow defines the computation. The workspace preserves the record of what was run. The asset layer keeps shared inputs and derived resources recoverable across runs.

## Start Here

If you want to see the system working before you learn the vocabulary, start in [Getting Started](getting-started/index.md). That section now follows the order most readers actually need. It begins with a runnable example, then slows down to explain how a workflow becomes a tracked run, and finally shows how CLI execution and `molcfg` profiles fit into the same model.

If you are evaluating the design rather than trying to run code immediately, begin with [Concepts](concept/index.md). Those pages explain the workflow model, the workspace model, the asset and reproducibility story, and the boundary between local execution and optional plugins.

## Documentation Structure

[Getting Started](getting-started/index.md) is the onboarding path. It is where you should go when the immediate question is "how do I make this work?" rather than "what does every type mean?" The pages there are intentionally practical and ordered like a first project rather than like an API surface.

[Concepts](concept/index.md) is for the mental model. Those pages explain what remains stable across scripts, profiles, runs, and transport layers. They are meant to clarify the system boundary, not enumerate every method.

[Guide](guide/index.md) is for deeper topics once the first-run path is already familiar. It is organized by theme: authoring workflows, working with persistent records and assets, and operating the server or scheduler bridge. [Development](development/index.md) remains the contributor-facing section for compiler internals, protocol design, and active specifications.

## Reading Path

The shortest practical route is [Quick Start](getting-started/quick-start.md), then [Track a Run](getting-started/tracked-runs.md), then [CLI and Profiles](getting-started/cli-and-profiles.md). Once that path makes sense, read [Workflow](concept/workflow.md) and [Assets and Reproducibility](concept/assets-and-reproducibility.md) to firm up the conceptual model behind the API you are already using.

## Runnable Examples

Every guide page has a matching stand-alone script under `examples/` — one per topic, cross-linked from the guide. Start at [`examples/README.md`](../examples/README.md) to see the full map.
