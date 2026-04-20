# Workspace

The workspace layer is the persistent model on disk. It exists so that a workflow definition does not have to carry every concern about provenance, execution history, and reusable data inside itself. The workflow says what should happen. The workspace records what was defined, what was attempted, and what was produced.

## The Persistent Hierarchy

MolExp stores work in a four-level hierarchy:

```text
Workspace
└── Project
    └── Experiment
        └── Run
```

A workspace is the root of a body of work. A project groups related work under that root. An experiment is a repeatable definition of one workflow together with one parameter set and one replica policy. A run is one concrete execution attempt under that experiment.

The important distinction is between experiment and run. An experiment is the definition you intend to repeat. A run is one realized attempt with a status, output files, error information, and execution history. Without that split, retries and comparisons quickly become ambiguous.

## Definition and Outcome

The workspace model is where MolExp insists on the difference between recipe and outcome. An experiment can be bound to a workflow source, a compiled workflow object, a parameter dictionary, replica settings, and provenance data such as the captured git commit. A run then carries the things that vary by execution: status, timestamps, profile metadata, results, logs, errors, and the per-attempt `ExecutionRecord` history.

That distinction is also why the Python API and the CLI behave slightly differently around run identity. In direct Python usage, `exp.run()` creates a fresh run unless you provide an explicit run id. In CLI usage, `molexp run` derives deterministic run ids from parameters, replica index, and profile metadata so that repeated invocations can find the same run again. The workspace layer supports both styles, but they should not be confused with each other.

## Profiles, Metadata, and Inspection

Profiles live at the boundary between workflow execution and workspace persistence. Tasks read the active configuration through `ctx.config`, but the chosen profile name, the merged config payload, and its `config_hash` are all stored on the run record. That makes configuration both executable and inspectable. You can look at `run.json` later and recover the exact config slice a run used, rather than relying on memory or shell history.

This is also the layer that the server and UI expose. Browsing projects, experiments, runs, and execution history is fundamentally browsing workspace state.

## Assets Belong Beside the Hierarchy

Reusable data is attached to the workspace layer through scoped asset libraries. Shared datasets may live at workspace or project scope. Experiment-specific derived resources may live with the experiment. Run-time code can register produced outputs into the experiment asset library so later runs can reuse them.

That asset story matters because a research record is not only source code and logs. It is also the set of concrete resources that later runs must still be able to find.

The next useful page depends on what still feels vague. If the open question is reusable data and provenance, continue with [Assets and Reproducibility](assets-and-reproducibility.md). If the open question is the concrete Python API, continue with [Workspace API](../guide/workspace-api.md).
