import { Ban, Copy, FileQuestion, FileText, Terminal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  DashboardCard,
  DashboardGrid,
  EmptyState,
  EntityPage,
  StatCard,
  StatGrid,
} from "@/app/components/entity";
import { listEntityTabs } from "@/app/registry";
import { formatDuration, formatScalar, statusTone } from "@/app/renderers/dashboardData";
import { RunExecutionsPanel } from "@/app/renderers/RunExecutionsPanel";
import { RunSnapshotPanel } from "@/app/renderers/SnapshotViewer";
import { workspaceApi } from "@/app/state/api";
import { useInspectedTask } from "@/app/state/inspectedTask";
import { useDiscoveredFileTypesForRun } from "@/app/state/useDiscoveredFileTypes";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAssetResponse, RendererProps } from "@/app/types";
import { useAlert, useConfirm } from "@/components/ConfirmDialog";
import { Button } from "@/components/ui/button";

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
  const duration = formatDuration(run.startedAt, run.finishedAt);
  const attemptCount = run.executionHistory.length;
  const groupedAssets = groupAssetsByKind(runAssets);
  const backend = run.executorInfo.backend || "local";

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

  // ── Overview: a compact dashboard. Numbers, I/O, errors — no execution
  // detail (that lives on its own tab now). ────────────────────────────────
  const overviewContent = (
    <DashboardGrid>
      <div className="lg:col-span-12">
        <StatGrid>
          <StatCard label="Status" value={run.status} tone={statusTone(run.status)} />
          <StatCard label="Duration" value={duration ?? "—"} muted={!duration} />
          <StatCard
            label="Attempts"
            value={attemptCount || 1}
            hint={attemptCount > 1 ? `${attemptCount} executions` : "single attempt"}
          />
          <StatCard label="Assets" value={runAssets.length} muted={runAssets.length === 0} />
          <StatCard label="Backend" value={backend} />
        </StatGrid>
      </div>

      <DashboardCard title="Parameters" className="lg:col-span-6">
        {parameterEntries.length === 0 ? (
          <p className="text-xs italic text-muted-foreground">No parameters recorded.</p>
        ) : (
          <dl className="grid gap-x-4 gap-y-2 sm:grid-cols-2">
            {parameterEntries.map(([key, value]) => (
              <div key={key} className="min-w-0">
                <dt className="truncate text-[11px] uppercase tracking-wide text-muted-foreground">
                  {key}
                </dt>
                <dd
                  className="truncate font-mono text-xs text-foreground"
                  title={formatScalar(value)}
                >
                  {formatScalar(value)}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </DashboardCard>

      <DashboardCard
        title="Results"
        className="lg:col-span-6"
        action={
          resultEntries.length > 0 ? (
            <span className="text-[11px] tabular-nums text-success">{resultEntries.length}</span>
          ) : undefined
        }
      >
        {resultEntries.length === 0 ? (
          <p className="text-xs italic text-muted-foreground">
            {run.status === "succeeded"
              ? "Finished without setting any result."
              : "Results appear after the run finishes."}
          </p>
        ) : (
          <dl className="grid gap-x-4 gap-y-2 sm:grid-cols-2">
            {resultEntries.map(([key, value]) => (
              <div key={key} className="min-w-0">
                <dt className="truncate text-[11px] uppercase tracking-wide text-muted-foreground">
                  {key}
                </dt>
                <dd
                  className="truncate font-mono text-xs text-foreground"
                  title={formatScalar(value)}
                >
                  {formatScalar(value)}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </DashboardCard>

      {run.errorMessage && (
        <DashboardCard title="Error" className="border-destructive/30 lg:col-span-12">
          <pre className="whitespace-pre-wrap break-words font-mono text-xs text-destructive">
            {run.errorMessage}
          </pre>
        </DashboardCard>
      )}

      <DashboardCard
        title="Generated assets"
        className="lg:col-span-7"
        bodyClassName={runAssets.length === 0 ? "p-3" : "p-0"}
        action={
          runAssets.length > 0 ? (
            <span className="text-[11px] tabular-nums text-muted-foreground">
              {runAssets.length}
            </span>
          ) : undefined
        }
      >
        {runAssets.length === 0 ? (
          <p className="text-xs italic text-muted-foreground">No assets registered for this run.</p>
        ) : (
          <div className="divide-y divide-border/50">
            {[...groupedAssets.entries()].map(([kind, assets]) => (
              <div key={kind}>
                <div className="flex items-center justify-between bg-muted/30 px-3 py-1 text-[11px] uppercase tracking-wide text-muted-foreground">
                  <span>{kind}</span>
                  <span>{assets.length}</span>
                </div>
                <ul className="divide-y divide-border/40">
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
      </DashboardCard>

      <DashboardCard title="Lineage" className="lg:col-span-5">
        <div className="flex flex-col gap-1.5">
          <Button
            variant="outline"
            size="sm"
            className="h-7 justify-start px-2 text-xs"
            onClick={() => setSelection({ objectType: "project", objectId: run.projectId })}
          >
            Project · {project?.name || run.projectId}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-7 justify-start px-2 text-xs"
            onClick={() => setSelection({ objectType: "experiment", objectId: run.experimentId })}
          >
            Experiment · {experiment?.name || run.experimentId}
          </Button>
          {workflow && (
            <Button
              variant="outline"
              size="sm"
              className="h-7 justify-start px-2 text-xs"
              onClick={() =>
                setSelection({
                  objectType: "workflow",
                  objectId: workflow.id,
                  workflowId: workflow.id,
                })
              }
            >
              Workflow · {workflow.name}
            </Button>
          )}
          <div className="mt-1 truncate font-mono text-[11px] text-muted-foreground" title={run.id}>
            run id: {run.id}
          </div>
        </div>
      </DashboardCard>
    </DashboardGrid>
  );

  // ── Executions: first-class tab (shared with the scheduler-backed run view).
  const executionsContent = (
    <RunExecutionsPanel
      run={run}
      workflow={workflow}
      selectedExecutionId={selectedExecutionId}
      onSelectExecution={setSelectedExecutionId}
      onInspectTask={inspectTask}
      onViewLogs={() => setActiveTab("logs")}
      onOpenWorkflow={
        workflow
          ? () =>
              setSelection({
                objectType: "workflow",
                objectId: workflow.id,
                workflowId: workflow.id,
              })
          : undefined
      }
    />
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
    {
      value: "executions",
      label: `Executions${attemptCount ? ` (${attemptCount})` : ""}`,
      content: executionsContent,
    },
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
