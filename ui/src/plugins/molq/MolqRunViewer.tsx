import { Ban, Boxes, Copy, FileQuestion, ServerCog, Terminal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  DashboardCard,
  DashboardGrid,
  EmptyState,
  EntityHeader,
  EntityMetric,
  EntityTabBar,
  EntityTabContent,
  EntityTabs,
  StatCard,
  StatGrid,
} from "@/app/components/entity";
import { listEntityTabs } from "@/app/registry";
import { formatDuration, formatScalar, statusTone } from "@/app/renderers/dashboardData";
import { RunExecutionsPanel } from "@/app/renderers/RunExecutionsPanel";
import { RunViewer } from "@/app/renderers/RunViewer";
import { workspaceApi } from "@/app/state/api";
import { useInspectedTask } from "@/app/state/inspectedTask";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";
import { useAlert, useConfirm } from "@/components/ConfirmDialog";
import { Button } from "@/components/ui/button";

const terminalRunStatuses = new Set(["succeeded", "failed", "cancelled", "skipped"]);

const getExecutorEntry = (
  executorInfo: Record<string, string>,
  ...keys: string[]
): string | null => {
  for (const key of keys) {
    const value = executorInfo[key];
    if (value) {
      return value;
    }
  }
  return null;
};

const formatExecutorLabel = (key: string): string => {
  return key.replace(/_/g, " ").replace(/\b\w/g, (match) => match.toUpperCase());
};

export const MolqRunViewer = (props: RendererProps): JSX.Element => {
  const { selection, snapshot, onRefresh } = props;
  const { setSelection } = useNavigationState(snapshot);
  const { inspectTask } = useInspectedTask();
  const [logs, setLogs] = useState<{ stdout?: string | null; stderr?: string | null } | null>(null);
  const [logsError, setLogsError] = useState<string | null>(null);
  const [selectedExecutionId, setSelectedExecutionId] = useState<string | null>(null);
  const runTabContributions = listEntityTabs("run");
  const { confirm, dialog: confirmDialog } = useConfirm();
  const { alert, dialog: alertDialog } = useAlert();

  const run = useMemo(() => {
    return snapshot.runs.find((item) => item.id === selection.objectId) ?? null;
  }, [selection.objectId, snapshot.runs]);

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

  if (run.executorInfo.backend !== "molq") {
    return <RunViewer {...props} />;
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

  const scheduler = getExecutorEntry(run.executorInfo, "scheduler") ?? "unknown";
  const cluster = getExecutorEntry(run.executorInfo, "cluster_name", "cluster") ?? "default";
  const jobId = getExecutorEntry(run.executorInfo, "job_id") ?? "pending";
  const schedulerJobId = getExecutorEntry(run.executorInfo, "scheduler_job_id") ?? "not assigned";
  const details = Object.entries(run.executorInfo);

  const duration = formatDuration(run.startedAt, run.finishedAt);
  const attemptCount = run.executionHistory.length;
  const parameterEntries = Object.entries(run.parameters ?? {});
  const resultEntries = Object.entries(run.results ?? {});

  const handleCopyRunId = (): void => {
    void navigator.clipboard.writeText(run.id);
  };

  const handleTabChange = (value: string): void => {
    setActiveTab(value);
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

  const fieldGrid = (entries: [string, unknown][], emptyLabel: string): JSX.Element =>
    entries.length === 0 ? (
      <p className="text-xs italic text-muted-foreground">{emptyLabel}</p>
    ) : (
      <dl className="grid gap-x-4 gap-y-2 sm:grid-cols-2">
        {entries.map(([key, value]) => (
          <div key={key} className="min-w-0">
            <dt className="truncate text-[11px] uppercase tracking-wide text-muted-foreground">
              {key}
            </dt>
            <dd className="truncate font-mono text-xs text-foreground" title={formatScalar(value)}>
              {formatScalar(value)}
            </dd>
          </div>
        ))}
      </dl>
    );

  return (
    <div className="flex h-full flex-col bg-background">
      <EntityHeader
        icon={ServerCog}
        title={run.name}
        status={run.status}
        subtitle={run.summary || undefined}
        metrics={
          <>
            <EntityMetric label="scheduler" value={scheduler} />
            <EntityMetric label="cluster" value={cluster} />
          </>
        }
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
      />

      <div className="flex flex-1 flex-col overflow-hidden">
        <EntityTabs value={activeTab} onValueChange={handleTabChange}>
          <EntityTabBar
            tabs={[
              { value: "overview", label: "Overview" },
              {
                value: "executions",
                label: `Executions${attemptCount ? ` (${attemptCount})` : ""}`,
              },
              { value: "logs", label: "Logs" },
              ...runTabContributions.map((tab) => ({ value: tab.value, label: tab.label })),
              { value: "scheduler", label: "Scheduler" },
            ]}
          />

          <EntityTabContent value="overview">
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
                  <StatCard label="Scheduler" value={scheduler} />
                  <StatCard label="Cluster" value={cluster} />
                </StatGrid>
              </div>

              <DashboardCard title="Scheduler" className="lg:col-span-5">
                <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <div className="min-w-0">
                    <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">
                      Job ID
                    </dt>
                    <dd className="truncate font-mono text-xs text-foreground">{jobId}</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">
                      Scheduler Job ID
                    </dt>
                    <dd className="truncate font-mono text-xs text-foreground">{schedulerJobId}</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">
                      Backend
                    </dt>
                    <dd className="truncate font-mono text-xs text-foreground">molq</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">
                      Cluster
                    </dt>
                    <dd className="truncate font-mono text-xs text-foreground">{cluster}</dd>
                  </div>
                </dl>
              </DashboardCard>

              <DashboardCard title="Lineage" className="lg:col-span-7">
                <div className="flex flex-wrap gap-1.5">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 px-2 text-xs"
                    onClick={() => setSelection({ objectType: "project", objectId: run.projectId })}
                  >
                    Project · {project?.name || run.projectId}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 px-2 text-xs"
                    onClick={() =>
                      setSelection({ objectType: "experiment", objectId: run.experimentId })
                    }
                  >
                    Experiment · {experiment?.name || run.experimentId}
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
                      Workflow · {workflow.name}
                    </Button>
                  )}
                </div>
              </DashboardCard>

              <DashboardCard title="Parameters" className="lg:col-span-6">
                {fieldGrid(parameterEntries, "No parameters recorded.")}
              </DashboardCard>

              <DashboardCard title="Results" className="lg:col-span-6">
                {fieldGrid(
                  resultEntries,
                  run.status === "succeeded"
                    ? "Finished without setting any result."
                    : "Results appear after the run finishes.",
                )}
              </DashboardCard>

              {run.errorMessage && (
                <DashboardCard title="Error" className="border-destructive/30 lg:col-span-12">
                  <pre className="whitespace-pre-wrap break-words font-mono text-xs text-destructive">
                    {run.errorMessage}
                  </pre>
                </DashboardCard>
              )}
            </DashboardGrid>
          </EntityTabContent>

          <EntityTabContent value="executions">
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
          </EntityTabContent>

          <EntityTabContent
            value="logs"
            className="m-0 flex flex-1 flex-col overflow-hidden bg-zinc-950 p-0 text-zinc-50 dark:bg-black"
          >
            <div className="flex items-center justify-between gap-2 border-b border-zinc-800 bg-zinc-900 px-3 py-1 font-mono text-[11px] text-zinc-400">
              <div className="flex items-center gap-2">
                <Terminal className="h-3 w-3" />
                <span>stdout/stderr</span>
                <span className="text-zinc-500">·</span>
                <span className="text-zinc-300">
                  {selectedExecutionId ? selectedExecutionId : "latest attempt"}
                </span>
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
          </EntityTabContent>

          {runTabContributions.map((tab) => {
            const TabComponent = tab.Component;
            return (
              <EntityTabContent key={tab.id} value={tab.value}>
                {activeTab === tab.value && <TabComponent key={selectedRunId} {...props} />}
              </EntityTabContent>
            );
          })}

          <EntityTabContent value="scheduler">
            <div className="flex-1 overflow-auto p-4">
              <section>
                <h3 className="flex items-center gap-1.5 text-[11px] font-medium uppercase text-muted-foreground">
                  <Boxes className="h-3.5 w-3.5" />
                  Normalized Executor Info
                </h3>
                <div className="mt-2 overflow-hidden border-y border-border/70">
                  <table className="w-full text-left text-sm">
                    <tbody className="divide-y divide-border/50">
                      {details.map(([key, value]) => (
                        <tr key={key}>
                          <td className="w-[220px] py-2 pr-4 text-xs font-medium text-muted-foreground">
                            {formatExecutorLabel(key)}
                          </td>
                          <td className="break-all py-2 font-mono text-xs text-foreground">
                            {value}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>
          </EntityTabContent>
        </EntityTabs>
      </div>
      {confirmDialog}
      {alertDialog}
    </div>
  );
};
