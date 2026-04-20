# Assets and Reproducibility

Reproducibility in MolExp is not only about being able to rerun Python code. It is about being able to recover the workflow definition, the chosen configuration, the persistent execution record, and the data or derived resources that the workflow depended on. The asset layer exists because those resources deserve first-class names and scopes instead of living as undocumented paths copied between scripts.

## Assets as Named Resources

MolExp represents every persistent byproduct — imported data, task artifacts, logs, checkpoints, captured exceptions, workflow execution state — as a typed `Asset` subclass scoped to the workspace, a project, an experiment, or a single run. External data enters through a `DataAssetLibrary` exposed as `ws.data_assets`, `project.data_assets`, or `exp.data_assets`. Run-time outputs attach to the run they were produced in through the `RunContext` accessors (`ctx.artifact.save(...)`, `ctx.log(name).append(...)`, `ctx.checkpoint(name, data=...)`). Every asset is indexed by a workspace-wide catalog so a single query can cross scopes.

This design keeps execution attempts transient while allowing useful outputs to become durable named resources. A task can later ask for `ctx.find_asset("training_data")` or `ctx.find_asset("feature-cache")` without hard-coding where those resources happen to live on disk, and tooling can ask the catalog for "all failed-run stack traces in experiment X" without walking the filesystem.

## Reproducibility Records

MolExp also persists the metadata that makes a run interpretable later. An experiment may carry the workflow source path and captured git commit. A run may carry a workflow snapshot reference, the selected profile name, the merged config payload, its `config_hash`, structured error information, and an execution history rather than a single flat status field. Together, those records are what turn a run directory into a scientific record rather than just a pile of output files.

The key idea is that reproducibility is distributed across several layers. The workflow defines the computation. The workspace gives it stable identity and execution history. The asset layer makes reusable data discoverable. None of those parts is sufficient by itself.

## FAIR-Oriented Boundaries

MolExp can support FAIR-oriented research practice inside a managed workspace, but it should not be oversold. Assets are findable within the workspace because they have stable names and metadata. They are accessible through the Python API, the server, and the UI. They become more interoperable when teams choose consistent metadata conventions. They become more reusable because workflows recover them by name and scope instead of through ephemeral local paths.

What MolExp does not do automatically is publish external identifiers, enforce a community schema, or turn local records into a repository-grade FAIR archive. It gives you a strong internal record and a coherent place to attach provenance. That is a substantial improvement over ad hoc folders, but it is not the same thing as full external FAIR publication.

## Why This Matters in Practice

Most workflow systems fail gradually. The script still exists, but nobody remembers which dataset directory was the real one, which profile was used for the paper figure, or whether the "final" checkpoint was generated before or after the last code change. MolExp tries to make those questions easier to answer by keeping the workflow, the run metadata, and the reusable assets in one persistent structure.
