# Molq Plugin and Cluster Submission

`molexp` can run workflows locally with no optional scheduler dependency. When you need SLURM, PBS, LSF, or another `molq`-supported backend, cluster submission is handled by the optional `submit_molq` plugin.

This page explains what that plugin is, how it fits into `molexp run`, and what responsibilities stay outside it.

## The Plugin's Role

In code, the cluster bridge lives in `molexp.plugins.submit_molq`. It is not a separate workflow system. It is an adapter that turns a `molexp` run into a `molq` job submission.

At runtime, `molexp run` loads this plugin only when you request a scheduler backend:

```bash
molexp run train.py --scheduler slurm --partition gpu --gpus 1 --cpus 8
molexp run train.py --scheduler pbs --queue batch
molexp run train.py --scheduler lsf --queue short
```

If `molq` is not installed, those commands fail fast with the install hint shown by the CLI:

```bash
pip install molq
```

## Optional Dependency Boundaries

The core `molexp` package is designed so that local workflows, workspace browsing, and most documentation examples do not require any scheduler client. That keeps the base environment small and prevents `import molexp` from failing on machines that do not have cluster tooling installed.

The `submit_molq` plugin exists precisely to keep scheduler integration outside the local core. You only pay that dependency cost when you ask for remote execution.

## Submission Flow

When `molexp run` is called with a scheduler backend, the control flow stays close to local execution until the point where individual runs are dispatched.

The high-level sequence is:

1. `molexp run` imports the workflow script and discovers workspaces via `me.entry(...)`.
2. It resolves the active `molcfg` profile and any `--override` values.
3. It scans projects and experiments, constructs or reuses eligible runs, and persists run metadata such as profile, config payload, config hash, and script path.
4. Instead of executing the workflow locally, it calls the `submit_molq` handler for each selected run.
5. That handler submits a worker command of the form `python -m molexp.cli execute <run_dir>` through `molq`.

The important takeaway is that the plugin does not invent a second execution model. It only changes where the worker process is launched.

## CLI Flags Become Scheduler Objects

The plugin turns `molexp` CLI flags into `molq` submission objects.

### Resource flags

These become `molq` resource settings:

- `--cpus`
- `--mem`
- `--gpus`
- `--gpu-type`
- `--time`

### Scheduling flags

These become scheduler placement settings:

- `--partition` or `--queue`
- `--account`
- `--qos`
- `--cluster`

`molexp` keeps the frontend flags scheduler-friendly, while `submit_molq` handles the translation into `molq`'s `JobResources`, `JobScheduling`, and `JobExecution` objects.

## Persisted Scheduler Metadata

After submission, `molexp` normalizes executor metadata and writes it back into the run metadata. That normalized payload includes fields such as:

- backend name (`molq`)
- scheduler name
- cluster name
- job ID
- scheduler job ID

This matters because the rest of the system should not have to know `molq`'s internal object model just to display status. The monitor, server responses, and UI can rely on a small normalized metadata shape instead.

## The Submitted Worker Command

The submitted job does not reimplement scheduling logic inside the cluster process. It runs:

```bash
python -m molexp.cli execute <run_dir>
```

That worker command reconstructs the `RunContext` from `run.json`, re-imports the original workflow script, finds the matching workflow binding for the run, rebuilds the `WorkflowSpec`, and then executes it against the existing run directory.

This is a key design point. Remote execution still goes through the same workflow and workspace abstractions as local execution:

- the same `WorkflowSpec`,
- the same `RunContext`,
- the same persisted profile config,
- the same `execution_history` model.

## Responsibilities That Stay Outside the Plugin

The `submit_molq` plugin is deliberately narrow in scope.

It does **not**:

- define task or actor behavior,
- change dependency semantics inside a workflow,
- replace the workspace hierarchy,
- interpret profile contents,
- or decide whether a run should be retried.

Those responsibilities remain in the workflow layer, workspace layer, and CLI orchestration logic. The plugin is a transport and metadata bridge, not a second runtime.

## Relationship to Monitoring and UI

When `molq` is installed, `molexp.plugins.discover_ui_plugins()` also exposes a frontend-facing `molq` UI plugin descriptor. That descriptor tells the UI layer that scheduler-aware run viewers and monitor surfaces are available.

On the CLI side, `molexp run --block` can immediately open the run monitor after submission, and `molexp explore` can later reopen the workspace explorer. Both rely on the normalized executor metadata that the plugin writes into the run metadata.

## Appropriate Use Cases

Use `submit_molq` when:

- the workflow is already working locally,
- you want scheduler-backed execution without rewriting the workflow,
- and you need `molexp` to persist cluster job identity alongside run metadata.

If you are still iterating on basic workflow semantics, local execution is usually the better place to start. Once the workflow shape is stable, the plugin lets you move that same shape onto a scheduler with minimal conceptual overhead.

## Runnable Example

`examples/operations/molq.py` composes the same `SubmitHandler` object the CLI would build for `--scheduler slurm` and prints the worker command plus the normalised `executor_info` payload — no cluster required.
