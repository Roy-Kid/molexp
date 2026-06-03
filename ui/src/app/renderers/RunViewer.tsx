import { ArrowRight, Ban, Code2, Copy, FileQuestion, FileText, Terminal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  EmptyState,
  EntityPage,
  KeyValueGrid,
  OverviewHighlight,
  OverviewHighlightGrid,
  OverviewPage,
  OverviewSection,
  StatusBadge,
} from "@/app/components/entity";
import { listEntityTabs } from "@/app/registry";
import { RunSnapshotPanel } from "@/app/renderers/SnapshotViewer";
import { parseWorkflowIr, WorkflowGraph } from "@/app/renderers/WorkflowGraph";
import { workspaceApi } from "@/app/state/api";
import { useInspectedTask } from "@/app/state/inspectedTask";
import { useDiscoveredFileTypesForRun } from "@/app/state/useDiscoveredFileTypes";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAssetResponse, RendererProps } from "@/app/types";
import { useAlert, useConfirm } from "@/components/ConfirmDialog";
import { Button } from "@/components/ui/button";

const formatScalar = (value: unknown): string => {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const formatDuration = (startIso: string | null, endIso: string | null): string | null => {
  if (!startIso || !endIso) return null;
  const start = Date.parse(startIso);
  const end = Date.parse(endIso);
  if (Number.isNaN(start) || Number.isNaN(end)) return null;
  const ms = Math.max(0, end - start);
  if (ms < 1000) return `${ms} ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds - minutes * 60);
  return `${minutes}m ${remainder}s`;
};

const formatTimeOfDay = (iso: string | null): string => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleTimeString();
};

const ASSET_KIND_ORDER = [
  "log",
  "execution_state",
  "artifact",
  "data",
  "checkpoint",
  "error_trace",
  "output",
];

const groupAssetsByKind = (assets: ApiAssetResponse[]): Map<string, ApiAssetResponse[]> => {
  const buckets = new Map<string, ApiAssetResponse[]>();
  for (const asset of assets) {
    const list = buckets.get(asset.kind) ?? [];
    list.push(asset);
    buckets.set(asset.kind, list);
  }
  // Stable order: known kinds first, unknown kinds alphabetical.
  return new Map(
    [...buckets.entries()].sort(([a], [b]) => {
      const ai = ASSET_KIND_ORDER.indexOf(a);
      const bi = ASSET_KIND_ORDER.indexOf(b);
      if (ai === -1 && bi === -1) return a.localeCompare(b);
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    }),
  );
};

const terminalRunStatuses = new Set(["succeeded", "failed", "cancelled", "skipped"]);

export const RunViewer = (props: RendererProps): JSX.Element => {
  const { selection, snapshot, onRefresh } = props;
  const { setSelection } = useNavigationState(snapshot);
  const { inspectTask } = useInspectedTask();
  const [logs, setLogs] = useState<{ stdout?: string | null; stderr?: string | null } | null>(null);
  const [logsError, setLogsError] = useState<string | null>(null);
  const [selectedExecutionId, setSelectedExecutionId] = useState<string | null>(null);
  const [runAssets, setRunAssets] = useState<ApiAssetResponse[]>([]);
  const runTabContributions = listEntityTabs("run");

  const { confirm, dialog: confirmDialog } = useConfirm();
  const { alert, dialog: alertDialog } = useAlert();

  const run = useMemo(() => {
    return snapshot.runs.find((r) => r.id === selection.objectId);
  }, [snapshot.runs, selection.objectId]);

  const runCoords = useMemo(
    () =>
      run ? { projectId: run.projectId, experimentId: run.experimentId, runId: run.id } : null,
    [run],
  );
  const { discovered: discoveredPlugins } = useDiscoveredFileTypesForRun(runCoords, "run");

  const requestedTab =
    selection.objectType === "run" ? (selection.objectView ?? "overview") : "overview";
  const selectedRunId = selection.objectId;
  const [activeTab, setActiveTab] = useState<string>(requestedTab);

  useEffect(() => {
    if (selectedRunId) {
      setActiveTab(requestedTab);
    }
  }, [requestedTab, selectedRunId]);

  // Capture IDs as primitives so the log effect's dep array doesn't churn
  // every poll — useWorkspaceState rebuilds snapshot.runs each tick.
  const runProjectId = run?.projectId;
  const runExperimentId = run?.experimentId;
  const runId = run?.id;

  useEffect(() => {
    let cancelled = false;
    setLogsError(null);

    if (!runId || !runProjectId || !runExperimentId || activeTab !== "logs") {
      return;
    }

    setLogs(null);
    const fetcher = selectedExecutionId
      ? workspaceApi.getRunExecutionLogs(runProjectId, runExperimentId, runId, selectedExecutionId)
      : workspaceApi.getRunLogs(runProjectId, runExperimentId, runId);

    fetcher
      .then((nextLogs) => {
        if (!cancelled) {
          setLogs(nextLogs);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setLogsError(error instanceof Error ? error.message : "Failed to load logs");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeTab, runProjectId, runExperimentId, runId, selectedExecutionId]);

  useEffect(() => {
    let cancelled = false;
    if (!run) {
      setRunAssets([]);
      return;
    }
    workspaceApi
      .getRunAssets(run.id)
      .then((assets) => {
        if (!cancelled) setRunAssets(assets);
      })
      .catch((err) => {
        console.warn(`Failed to load assets for run ${run.id}:`, err);
        if (!cancelled) setRunAssets([]);
      });
    return () => {
      cancelled = true;
    };
  }, [run]);

  if (!run) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          icon={<FileQuestion className="h-6 w-6" />}
          title="Run not found"
          description="It may have been deleted or not yet synced."
        />
      </div>
    );
  }

  const project = snapshot.projects.find((item) => item.id === run.projectId);
  const experiment = snapshot.experiments.find((item) => item.id === run.experimentId);
  const workflow = experiment
    ? snapshot.workflows.find(
        (item) =>
          item.experimentId === experiment.id &&
          (item.name === experiment.workflowFile || item.id === experiment.workflowFile),
      )
    : undefined;

  const parameterEntries = Object.entries(run.parameters ?? {});
  const resultEntries = Object.entries(run.results ?? {});
  const workflowIr = parseWorkflowIr(run.workflowSource);
  const duration = formatDuration(run.startedAt, run.finishedAt);
  const attemptCount = run.executionHistory.length;
  const groupedAssets = groupAssetsByKind(runAssets);

  const handleCopyRunId = (): void => {
    void navigator.clipboard.writeText(run.id);
  };

  const handleCancelRun = async (): Promise<void> => {
    if (terminalRunStatuses.has(run.status)) return;
    const confirmed = await confirm({
      title: "Mark run as cancelled?",
      description: (
        <>
          Run <code className="rounded bg-muted px-1 py-0.5 text-xs">{run.id}</code> will be marked
          cancelled in the workspace. This does not stop any underlying scheduler job.
        </>
      ),
      confirmLabel: "Mark cancelled",
      destructive: true,
    });
    if (!confirmed) return;
    try {
      await workspaceApi.updateRunStatus(run.projectId, run.experimentId, run.id, "cancelled");
      onRefresh();
    } catch (error) {
      console.error("Failed to mark run cancelled:", error);
      void alert({
        title: "Failed to mark run cancelled",
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const overviewContent = (
    <OverviewPage
      aside={
        <>
          <OverviewSection title="Highlights">
            <OverviewHighlightGrid>
              <OverviewHighlight
                label="Started"
                value={
                  run.startedAt ? (
                    new Date(run.startedAt).toLocaleString()
                  ) : (
                    <span className="text-muted-foreground">—</span>
                  )
                }
              />
              <OverviewHighlight
                label="Duration"
                value={duration ?? <span className="text-muted-foreground">—</span>}
                detail={
                  run.finishedAt
                    ? `Finished ${new Date(run.finishedAt).toLocaleTimeString()}`
                    : undefined
                }
              />
              <OverviewHighlight
                label="Attempts"
                value={attemptCount || 1}
                detail={attemptCount > 1 ? `${attemptCount} executions` : undefined}
              />
              <OverviewHighlight
                label="Generated"
                value={runAssets.length}
                detail={runAssets.length === 1 ? "asset" : "assets"}
              />
              <OverviewHighlight label="Backend" value={run.executorInfo.backend || "local"} />
              <OverviewHighlight
                label="Run ID"
                value={<span className="font-mono text-sm">{run.id}</span>}
              />
            </OverviewHighlightGrid>
          </OverviewSection>

          <OverviewSection title="Relationships">
            <div className="flex flex-wrap gap-1.5">
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={() => setSelection({ objectType: "project", objectId: run.projectId })}
              >
                Project: {project?.name || run.projectId}
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={() =>
                  setSelection({ objectType: "experiment", objectId: run.experimentId })
                }
              >
                Experiment: {experiment?.name || run.experimentId}
              </Button>
              {workflow && (
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 px-2 text-xs"
                  onClick={() =>
                    setSelection({
                      objectType: "workflow",
                      objectId: workflow.id,
                      workflowId: workflow.id,
                    })
                  }
                >
                  Workflow: {workflow.name}
                </Button>
              )}
            </div>
          </OverviewSection>
        </>
      }
    >
      <OverviewSection title="Inputs → Outputs">
        <div className="grid gap-4 md:grid-cols-[1fr_auto_1fr]">
          <div className="rounded-md border border-border/70 bg-muted/30 p-3">
            <div className="mb-2 flex items-center gap-2 text-[11px] uppercase tracking-wide text-muted-foreground">
              <span>Parameters</span>
              <span className="text-foreground/60">({parameterEntries.length})</span>
            </div>
            {parameterEntries.length === 0 ? (
              <p className="text-xs italic text-muted-foreground">No parameters recorded.</p>
            ) : (
              <KeyValueGrid
                items={parameterEntries.map(([key, value]) => ({
                  label: key,
                  value: <span className="font-mono text-xs">{formatScalar(value)}</span>,
                }))}
              />
            )}
          </div>
          <div className="hidden items-center justify-center text-muted-foreground md:flex">
            <ArrowRight className="h-5 w-5" />
          </div>
          <div
            className={`rounded-md border p-3 ${
              resultEntries.length > 0
                ? "border-emerald-500/30 bg-emerald-500/5"
                : "border-border/70 bg-muted/30"
            }`}
          >
            <div className="mb-2 flex items-center gap-2 text-[11px] uppercase tracking-wide text-muted-foreground">
              <span>Results</span>
              <span className="text-foreground/60">({resultEntries.length})</span>
            </div>
            {resultEntries.length === 0 ? (
              <p className="text-xs italic text-muted-foreground">
                {run.status === "succeeded"
                  ? "Workflow finished without setting any result."
                  : "Results appear after the run finishes."}
              </p>
            ) : (
              <KeyValueGrid
                items={resultEntries.map(([key, value]) => ({
                  label: key,
                  value: (
                    <span className="font-mono text-xs text-foreground">{formatScalar(value)}</span>
                  ),
                }))}
              />
            )}
          </div>
        </div>
      </OverviewSection>

      {run.errorMessage && (
        <OverviewSection title="Error">
          <div className="rounded-md border border-rose-500/30 bg-rose-500/5 p-3">
            <pre className="whitespace-pre-wrap break-words font-mono text-xs text-rose-700 dark:text-rose-300">
              {run.errorMessage}
            </pre>
          </div>
        </OverviewSection>
      )}

      <OverviewSection title="What ran">
        {workflowIr ? (
          <div className="max-w-4xl space-y-2">
            <WorkflowGraph
              ir={workflowIr}
              height={460}
              onNodeClick={(taskId) => inspectTask(taskId, run.id)}
            />
            <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground">
              {workflowIr.name && <span className="font-mono">{workflowIr.name}</span>}
              <span>
                {workflowIr.task_configs.length} tasks · {workflowIr.links.length} dependencies
              </span>
              {run.configHash && (
                <span className="font-mono">config: {run.configHash.slice(0, 12)}</span>
              )}
              {run.profile && <span>profile: {run.profile}</span>}
              {workflow && (
                <button
                  type="button"
                  className="inline-flex items-center gap-1 underline-offset-2 hover:text-foreground hover:underline"
                  onClick={() =>
                    setSelection({
                      objectType: "workflow",
                      objectId: workflow.id,
                      workflowId: workflow.id,
                    })
                  }
                >
                  Open workflow <ArrowRight className="h-3 w-3" />
                </button>
              )}
            </div>
          </div>
        ) : run.workflowSource ? (
          <button
            type="button"
            className="flex w-full max-w-3xl items-start gap-3 rounded-md border border-border/70 bg-muted/30 p-3 text-left transition-colors hover:border-border hover:bg-muted/50"
            onClick={() => {
              if (workflow) {
                setSelection({
                  objectType: "workflow",
                  objectId: workflow.id,
                  workflowId: workflow.id,
                });
              }
            }}
            disabled={!workflow}
            title={workflow ? "Open workflow" : "Workflow object not loaded"}
          >
            <Code2 className="mt-0.5 h-4 w-4 flex-none text-muted-foreground" />
            <div className="min-w-0 flex-1">
              <div className="break-all font-mono text-xs text-foreground">
                {run.workflowSource}
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground">
                {run.configHash && (
                  <span className="font-mono">config: {run.configHash.slice(0, 12)}</span>
                )}
                {run.profile && <span>profile: {run.profile}</span>}
              </div>
            </div>
          </button>
        ) : (
          <p className="text-sm italic text-muted-foreground">
            No workflow snapshot recorded for this run.
          </p>
        )}
      </OverviewSection>

      <OverviewSection title="Timeline">
        {run.executionHistory.length === 0 ? (
          <p className="text-sm italic text-muted-foreground">No execution attempts recorded.</p>
        ) : (
          <ol className="divide-y divide-border/70 overflow-hidden rounded-md border border-border/70">
            {run.executionHistory.map((rec, index) => {
              const isSelected = selectedExecutionId === rec.executionId;
              return (
                <li key={rec.executionId}>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedExecutionId(rec.executionId);
                      setActiveTab("logs");
                    }}
                    className={`grid w-full grid-cols-[auto_minmax(0,1fr)_auto_auto] items-center gap-3 px-3 py-2 text-left text-xs transition-colors hover:bg-muted/40 ${
                      isSelected ? "bg-muted/60 ring-1 ring-inset ring-primary/40" : ""
                    }`}
                    title="View logs for this attempt"
                  >
                    <span className="font-mono text-muted-foreground">#{index + 1}</span>
                    <span className="truncate font-mono text-foreground" title={rec.executionId}>
                      {rec.executionId}
                    </span>
                    <span className="text-muted-foreground">
                      {formatTimeOfDay(rec.startedAt)}
                      {rec.finishedAt ? ` → ${formatTimeOfDay(rec.finishedAt)}` : ""}
                      {(() => {
                        const d = formatDuration(rec.startedAt, rec.finishedAt);
                        return d ? ` · ${d}` : "";
                      })()}
                    </span>
                    <StatusBadge status={rec.status} size="sm" />
                  </button>
                </li>
              );
            })}
          </ol>
        )}
      </OverviewSection>

      <OverviewSection title="Generated">
        {runAssets.length === 0 ? (
          <p className="text-sm italic text-muted-foreground">
            No assets were registered for this run.
          </p>
        ) : (
          <div className="space-y-3">
            {[...groupedAssets.entries()].map(([kind, assets]) => (
              <div key={kind} className="rounded-md border border-border/70">
                <div className="flex items-center justify-between border-b border-border/70 bg-muted/30 px-3 py-1.5 text-[11px] uppercase tracking-wide text-muted-foreground">
                  <span>{kind}</span>
                  <span>{assets.length}</span>
                </div>
                <ul className="divide-y divide-border/70">
                  {assets.map((asset) => (
                    <li key={asset.id}>
                      <button
                        type="button"
                        className="flex w-full items-center justify-between gap-3 px-3 py-1.5 text-left text-xs transition-colors hover:bg-muted/40"
                        onClick={() => setSelection({ objectType: "asset", objectId: asset.id })}
                      >
                        <span className="truncate font-medium text-foreground" title={asset.name}>
                          {asset.name}
                        </span>
                        <span
                          className="truncate font-mono text-[11px] text-muted-foreground"
                          title={asset.path}
                        >
                          {asset.path}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}
      </OverviewSection>
    </OverviewPage>
  );

  const selectedExecutionIndex = selectedExecutionId
    ? run.executionHistory.findIndex((rec) => rec.executionId === selectedExecutionId)
    : -1;
  const attemptCountForLabel = run.executionHistory.length;
  let attemptLabel: string;
  if (selectedExecutionIndex >= 0) {
    attemptLabel = `attempt #${selectedExecutionIndex + 1} (${selectedExecutionId})`;
  } else if (attemptCountForLabel > 0) {
    attemptLabel = `latest attempt (#${attemptCountForLabel})`;
  } else {
    attemptLabel = "latest attempt";
  }
  const logsContent = (
    <div className="flex h-full flex-1 flex-col overflow-hidden bg-zinc-950 text-zinc-50 dark:bg-black">
      <div className="flex items-center justify-between gap-2 border-b border-zinc-800 bg-zinc-900 px-3 py-1 font-mono text-[11px] text-zinc-400">
        <div className="flex items-center gap-2">
          <Terminal className="h-3 w-3" />
          <span>stdout/stderr</span>
          <span className="text-zinc-500">·</span>
          <span className="text-zinc-300">{attemptLabel}</span>
        </div>
        {selectedExecutionId && (
          <button
            type="button"
            className="text-zinc-400 underline-offset-2 hover:text-zinc-100 hover:underline"
            onClick={() => setSelectedExecutionId(null)}
          >
            view latest
          </button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-3 font-mono text-xs">
        {logsError ? (
          <div className="text-rose-300">{logsError}</div>
        ) : logs ? (
          <div className="space-y-4">
            <section>
              <div className="mb-1 text-[11px] uppercase text-zinc-500">stdout</div>
              <pre className="whitespace-pre-wrap text-zinc-100">
                {logs.stdout || "No stdout captured."}
              </pre>
            </section>
            <section>
              <div className="mb-1 text-[11px] uppercase text-zinc-500">stderr</div>
              <pre className="whitespace-pre-wrap text-rose-100">
                {logs.stderr || "No stderr captured."}
              </pre>
            </section>
          </div>
        ) : (
          <div className="italic opacity-60">Loading logs...</div>
        )}
      </div>
    </div>
  );

  const tabs = [
    { value: "overview", label: "Overview", content: overviewContent },
    { value: "logs", label: "Logs", content: logsContent },
    ...runTabContributions.map((tab) => {
      const TabComponent = tab.Component;
      return {
        value: tab.value,
        label: tab.label,
        content: activeTab === tab.value ? <TabComponent key={selectedRunId} {...props} /> : null,
      };
    }),
    ...discoveredPlugins.map(({ contribution, files }) => {
      const PluginComponent = contribution.Component;
      return {
        value: contribution.value,
        label: `${contribution.label} (${files.length})`,
        content:
          activeTab === contribution.value ? (
            <PluginComponent key={selectedRunId} {...props} discoveredFiles={files} />
          ) : null,
      };
    }),
    { value: "snapshot", label: "Snapshot", content: <RunSnapshotPanel run={run} /> },
  ];

  return (
    <>
      <EntityPage
        icon={FileText}
        title={run.name}
        status={run.status}
        subtitle={run.summary || undefined}
        actions={
          <>
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={handleCopyRunId}>
              <Copy className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-muted-foreground hover:text-destructive"
              disabled={terminalRunStatuses.has(run.status)}
              title="Updates workspace status only; it does not cancel a scheduler job."
              onClick={() => {
                void handleCancelRun();
              }}
            >
              <Ban className="h-4 w-4" />
            </Button>
          </>
        }
        activeTab={activeTab}
        onActiveTabChange={setActiveTab}
        tabs={tabs}
      />
      {confirmDialog}
      {alertDialog}
    </>
  );
};
