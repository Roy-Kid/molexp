import { Ban, Boxes, Copy, FileQuestion, ServerCog } from "lucide-react";
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
import { formatScalar, statusTone } from "@/app/renderers/dashboardData";
import { RunExecutionsPanel } from "@/app/renderers/RunExecutionsPanel";
import { RunLogsPanel } from "@/app/renderers/RunLogsPanel";
import { RunViewer } from "@/app/renderers/RunViewer";
import { useRunViewer } from "@/app/renderers/useRunViewer";
import { RunMetricsView } from "@/app/runs/metrics/RunMetricsView";
import type { RendererProps } from "@/app/types";
import { Button } from "@/components/ui/button";

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

  const scheduler = getExecutorEntry(run.executorInfo, "scheduler") ?? "unknown";
  const cluster = getExecutorEntry(run.executorInfo, "cluster_name", "cluster") ?? "default";
  const jobId = getExecutorEntry(run.executorInfo, "job_id") ?? "pending";
  const schedulerJobId = getExecutorEntry(run.executorInfo, "scheduler_job_id") ?? "not assigned";
  const details = Object.entries(run.executorInfo);

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
      />

      <div className="flex flex-1 flex-col overflow-hidden">
        <EntityTabs value={activeTab} onValueChange={setActiveTab}>
          <EntityTabBar
            tabs={[
              { value: "overview", label: "Overview" },
              {
                value: "executions",
                label: `Executions${attemptCount ? ` (${attemptCount})` : ""}`,
              },
              { value: "logs", label: "Logs" },
              { value: "metrics", label: "Metrics" },
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
            <RunLogsPanel
              logs={logs}
              logsError={logsError}
              selectedExecutionId={selectedExecutionId}
              attemptLabel={selectedExecutionId ? selectedExecutionId : "latest attempt"}
              onViewLatest={() => setSelectedExecutionId(null)}
            />
          </EntityTabContent>

          <EntityTabContent value="metrics">
            {activeTab === "metrics" && (
              <RunMetricsView
                key={run.id}
                projectId={run.projectId}
                experimentId={run.experimentId}
                runId={run.id}
              />
            )}
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
