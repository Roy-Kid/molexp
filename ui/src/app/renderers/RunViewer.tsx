import { Ban, Copy, FileQuestion, PlayCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { DashboardCard, DashboardGrid, EmptyState, EntityPage } from "@/app/components/entity";
import { formatScalar } from "@/app/renderers/dashboardData";
import { RunExecutionsPanel } from "@/app/renderers/RunExecutionsPanel";
import { RunLogsPanel } from "@/app/renderers/RunLogsPanel";
import { useRunViewer } from "@/app/renderers/useRunViewer";
import { RunMetricsView } from "@/app/runs/metrics/RunMetricsView";
import { RunActions } from "@/app/runs/RunActions";
import { workspaceApi } from "@/app/state/api";
import { useDiscoveredFileTypesForRun } from "@/app/state/useDiscoveredFileTypes";
import type { ApiAssetResponse, RendererProps } from "@/app/types";
import { Button } from "@/components/ui/button";

export const RunViewer = (props: RendererProps): JSX.Element => {
  const {
    run,
    project,
    experiment,
    workflow,
    selectedRunId,
    activeTab,
    setActiveTab,
    logs,
    logsError,
    selectedExecutionId,
    setSelectedExecutionId,
    duration,
    attemptCount,
    parameterEntries,
    resultEntries,
    isTerminal,
    runTabContributions,
    inspectTask,
    setSelection,
    handleCopyRunId,
    handleCancelRun,
    confirmDialog,
    alertDialog,
  } = useRunViewer(props);

  const [runAssets, setRunAssets] = useState<ApiAssetResponse[]>([]);

  const runCoords = useMemo(
    () =>
      run ? { projectId: run.projectId, experimentId: run.experimentId, runId: run.id } : null,
    [run],
  );
  const { discovered: discoveredPlugins } = useDiscoveredFileTypesForRun(runCoords, "run");

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

  const backend = run.executorInfo.backend || "local";

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
    <div className="flex h-full flex-1 flex-col overflow-hidden bg-background text-foreground">
      <RunLogsPanel
        logs={logs}
        logsError={logsError}
        selectedExecutionId={selectedExecutionId}
        attemptLabel={attemptLabel}
        onViewLatest={() => setSelectedExecutionId(null)}
      />
    </div>
  );

  // Only surface the Logs tab when there is actually something to show — a run
  // that captured no stdout/stderr/run.log shouldn't carry an empty tab.
  const hasLogs = Boolean(logs?.stdout || logs?.stderr);
  const tabs = [
    { value: "overview", label: "Overview", content: overviewContent },
    {
      value: "executions",
      label: `Executions${attemptCount ? ` (${attemptCount})` : ""}`,
      content: executionsContent,
    },
    ...(hasLogs ? [{ value: "logs", label: "Logs", content: logsContent }] : []),
    {
      // First-class Metrics tab — render the molplot metrics view from the
      // resolved run's coords directly (run is non-null past the not-found
      // guard), rather than relying on the file-type discovery path.
      value: "metrics",
      label: "Metrics",
      content:
        activeTab === "metrics" ? (
          <RunMetricsView
            key={run.id}
            projectId={run.projectId}
            experimentId={run.experimentId}
            runId={run.id}
          />
        ) : null,
    },
    ...runTabContributions.map((tab) => {
      const TabComponent = tab.Component;
      return {
        value: tab.value,
        label: tab.label,
        content: activeTab === tab.value ? <TabComponent key={selectedRunId} {...props} /> : null,
      };
    }),
    // Drop a discovered "metrics" contribution — the first-class Metrics tab
    // above owns it, avoiding a duplicate if discovery ever fires.
    ...discoveredPlugins
      .filter(({ contribution }) => contribution.value !== "metrics")
      .map(({ contribution, files }) => {
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
            <RunActions
              projectId={run.projectId}
              experimentId={run.experimentId}
              runId={run.id}
              status={run.status}
              params={run.parameters ?? {}}
              onChanged={props.onRefresh}
            />
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={handleCopyRunId}>
              <Copy className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-muted-foreground hover:text-destructive"
              disabled={isTerminal}
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
