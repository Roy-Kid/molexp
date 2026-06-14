import { Ban, Copy, FileQuestion, PlayCircle, Terminal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { DashboardCard, DashboardGrid, EmptyState, EntityPage } from "@/app/components/entity";
import { listEntityTabs } from "@/app/registry";
import { formatDuration, formatScalar } from "@/app/renderers/dashboardData";
import { RunExecutionsPanel } from "@/app/renderers/RunExecutionsPanel";
import { workspaceApi } from "@/app/state/api";
import { useInspectedTask } from "@/app/state/inspectedTask";
import { useDiscoveredFileTypesForRun } from "@/app/state/useDiscoveredFileTypes";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAssetResponse, RendererProps } from "@/app/types";
import { useAlert, useConfirm } from "@/components/ConfirmDialog";
import { Button } from "@/components/ui/button";

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

  const overviewContent = (
    <DashboardGrid>
      <DashboardCard title="Run details" className="lg:col-span-8">
        <dl className="grid gap-x-6 gap-y-3 md:grid-cols-2">
          <div className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Run ID
            </dt>
            <dd className="mt-0.5 truncate font-mono text-xs text-foreground">{run.id}</dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Project
            </dt>
            <dd className="mt-0.5 truncate text-sm text-foreground">
              {project?.name ?? run.projectId}
            </dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Experiment
            </dt>
            <dd className="mt-0.5 truncate text-sm text-foreground">
              {experiment?.name ?? run.experimentId}
            </dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Updated
            </dt>
            <dd className="mt-0.5 truncate text-sm text-foreground">
              {new Date(run.updatedAt).toLocaleString()}
            </dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Started
            </dt>
            <dd className="mt-0.5 truncate text-sm text-foreground">
              {run.startedAt ? new Date(run.startedAt).toLocaleString() : "—"}
            </dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Finished
            </dt>
            <dd className="mt-0.5 truncate text-sm text-foreground">
              {run.finishedAt ? new Date(run.finishedAt).toLocaleString() : "—"}
            </dd>
          </div>
          {run.summary && (
            <div className="min-w-0 md:col-span-2">
              <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                Summary
              </dt>
              <dd className="mt-0.5 text-sm leading-6 text-foreground">{run.summary}</dd>
            </div>
          )}
        </dl>
      </DashboardCard>

      <DashboardCard title="Run stats" className="lg:col-span-4">
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
          <div>
            <dt className="text-muted-foreground">Duration</dt>
            <dd className="font-semibold tabular-nums text-foreground">{duration ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Attempts</dt>
            <dd className="font-semibold tabular-nums text-foreground">{attemptCount || 1}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Backend</dt>
            <dd className="truncate font-semibold text-foreground">{backend}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Assets</dt>
            <dd className="font-semibold tabular-nums text-foreground">{runAssets.length}</dd>
          </div>
        </dl>
      </DashboardCard>

      {run.errorMessage && (
        <DashboardCard title="Error" className="border-destructive/30 lg:col-span-12">
          <pre className="whitespace-pre-wrap break-words font-mono text-xs text-destructive">
            {run.errorMessage}
          </pre>
        </DashboardCard>
      )}

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

      <DashboardCard title="Results" className="lg:col-span-6">
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
  ];

  return (
    <>
      <EntityPage
        icon={PlayCircle}
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
