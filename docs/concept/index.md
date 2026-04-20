# Concepts

MolExp becomes much easier to read once you stop treating it as one large framework and start treating it as a few connected models with different responsibilities. The workflow model describes the computation itself. The workspace model describes the persistent scientific record on disk. The plugin layer describes optional transport or integration points. Assets and reproducibility cut across those layers because they are the reason the separation exists in the first place.

## The System Model

The workflow layer answers the question "what should happen?" It gives you tasks, actors, dependency edges, compilation, and execution. The workspace layer answers the question "what survives after execution?" It gives you projects, experiments, runs, execution history, and stored metadata. Plugins answer the question "where should execution happen?" without replacing either the workflow semantics or the workspace record.

Those boundaries are not academic. They are what make it possible for one workflow definition to run locally during development, then under a tracked run in a workspace, then from a cluster worker launched by the CLI, all without changing what the workflow itself means.

## Why the Split Matters

Research code usually becomes hard to trust for a boring reason: the script, the parameters, the logs, the outputs, and the "real" dataset path all drift apart. MolExp tries to prevent that drift by making each kind of state live in an explicit place. A workflow definition remains a reusable graph. An experiment remains the repeatable definition attached to a parameter set. A run remains one concrete execution attempt. An asset remains a named resource at a defined scope instead of an undocumented absolute path copied from a notebook.

Once you read the system this way, the rest of the documentation stops looking like a pile of unrelated APIs. Each page is really about one of those boundaries.

## Reading This Section

[Workflow](workflow.md) explains the computation model: what a `WorkflowSpec` is, why authoring and execution are separate, and what stays outside the workflow layer. [Workspace](workspace.md) explains the persistent hierarchy on disk and the difference between an experiment definition and a run. [Assets and Reproducibility](assets-and-reproducibility.md) explains why reusable data lives in scoped libraries and what MolExp can honestly claim about FAIR-style records. [Plugins](plugins.md) explains the optional transport layer that reaches beyond local execution.
