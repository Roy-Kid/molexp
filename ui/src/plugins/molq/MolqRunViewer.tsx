import { Boxes, FileQuestion, ServerCog } from "lucide-react";
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
import { formatScalar, statusTone } from "@/app/renderers/dashboardData";
import { RunExecutionsPanel } from "@/app/renderers/RunExecutionsPanel";
import { RunLogsPanel } from "@/app/renderers/RunLogsPanel";
import { RunViewer } from "@/app/renderers/RunViewer";
import { RunOutputsPanel } from "@/app/renderers/run/RunOutputsPanel";
import { useRunViewer } from "@/app/renderers/useRunViewer";
import { POST_DISPATCH_TAB, RunToolbar } from "@/app/runs/RunToolbar";
import { workspaceApi } from "@/app/state/api";
import { useDiscoveredFileTypesForRun } from "@/app/state/useDiscoveredFileTypes";
import type { ApiAssetResponse, RendererProps } from "@/app/types";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { WorkbenchAction } from "@/components/workbench";

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
    runTabContributions,
    inspectTask,
    setSelection,
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
      .catch(() => {
        if (!cancelled) setRunAssets([]);
      });
    return () => {
      cancelled = true;
    };
  }, [run]);

  if (!run) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState icon={<FileQuestion className="h-6 w-6" />} title="Not found" />
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
  const outputResults = resultEntries.map(([key, value]) => ({ key, value }));

  const fieldGrid = (entries: [string, unknown][], emptyLabel: string): JSX.Element =>
    entries.length === 0 ? (
      <p className="text-label italic text-muted-foreground">{emptyLabel}</p>
    ) : (
      <dl className="grid gap-x-4 gap-y-2 sm:grid-cols-2">
        {entries.map(([key, value]) => (
          <div key={key} className="min-w-0">
            <dt className="truncate text-micro uppercase tracking-wide text-muted-foreground">
              {key}
            </dt>
            <dd
              className="truncate font-mono text-label text-foreground"
              title={formatScalar(value)}
            >
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
          <RunToolbar
            projectId={run.projectId}
            experimentId={run.experimentId}
            runId={run.id}
            status={run.status}
            params={run.parameters ?? {}}
            onRefresh={props.onRefresh}
            onCancel={handleCancelRun}
            onDispatched={() => setActiveTab(POST_DISPATCH_TAB)}
            onOpenAgent={() =>
              setSelection({
                objectType: "agent",
                objectId: "new",
                scope: {
                  projectId: run.projectId,
                  experimentId: run.experimentId,
                  runId: run.id,
                },
              })
            }
            onHarvested={(path) => {
              props.onRefresh();
              if (path) {
                setSelection({ objectType: "knowledge", objectId: path });
              }
            }}
          />
        }
      />

      <div className="flex flex-1 flex-col overflow-hidden">
        <EntityTabs value={activeTab} onValueChange={setActiveTab}>
          <EntityTabBar
            tabs={[
              { value: "overview", label: "Overview" },
              {
                value: "outputs",
                label:
                  runAssets.length + resultEntries.length > 0
                    ? `Outputs (${runAssets.length + resultEntries.length})`
                    : "Outputs",
              },
              {
                value: "executions",
                label: `Executions${attemptCount ? ` (${attemptCount})` : ""}`,
              },
              { value: "logs", label: "Logs" },
              ...runTabContributions.map((tab) => ({ value: tab.value, label: tab.label })),
              ...discoveredPlugins.map(({ contribution, files }) => ({
                value: contribution.value,
                label: `${contribution.label} (${files.length})`,
              })),
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
                    <dt className="text-micro uppercase tracking-wide text-muted-foreground">
                      Job ID
                    </dt>
                    <dd className="truncate font-mono text-label text-foreground">{jobId}</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-micro uppercase tracking-wide text-muted-foreground">
                      Scheduler Job ID
                    </dt>
                    <dd className="truncate font-mono text-label text-foreground">
                      {schedulerJobId}
                    </dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-micro uppercase tracking-wide text-muted-foreground">
                      Backend
                    </dt>
                    <dd className="truncate font-mono text-label text-foreground">molq</dd>
                  </div>
                  <div className="min-w-0">
                    <dt className="text-micro uppercase tracking-wide text-muted-foreground">
                      Cluster
                    </dt>
                    <dd className="truncate font-mono text-label text-foreground">{cluster}</dd>
                  </div>
                </dl>
              </DashboardCard>

              <DashboardCard title="Lineage" className="lg:col-span-7">
                <div className="flex flex-wrap gap-2">
                  <WorkbenchAction
                    kind="secondary"
                    size="compact"
                    className="h-control-compact px-2 text-label"
                    onClick={() => setSelection({ objectType: "project", objectId: run.projectId })}
                  >
                    Project · {project?.name || run.projectId}
                  </WorkbenchAction>
                  <WorkbenchAction
                    kind="secondary"
                    size="compact"
                    className="h-control-compact px-2 text-label"
                    onClick={() =>
                      setSelection({ objectType: "experiment", objectId: run.experimentId })
                    }
                  >
                    Experiment · {experiment?.name || run.experimentId}
                  </WorkbenchAction>
                  {workflow && (
                    <WorkbenchAction
                      kind="secondary"
                      size="compact"
                      className="h-control-compact px-2 text-label"
                      onClick={() =>
                        setSelection({
                          objectType: "workflow",
                          objectId: workflow.id,
                          workflowId: workflow.id,
                        })
                      }
                    >
                      Workflow · {workflow.name}
                    </WorkbenchAction>
                  )}
                </div>
              </DashboardCard>

              {run.errorMessage && (
                <DashboardCard title="Error" className="border-destructive/30 lg:col-span-12">
                  <pre className="whitespace-pre-wrap break-words font-mono text-label text-destructive">
                    {run.errorMessage}
                  </pre>
                </DashboardCard>
              )}

              <DashboardCard title="Parameters" className="lg:col-span-6">
                {fieldGrid(parameterEntries, "—")}
              </DashboardCard>

              <DashboardCard title="Results" className="lg:col-span-6">
                {fieldGrid(resultEntries, "—")}
              </DashboardCard>
            </DashboardGrid>
          </EntityTabContent>

          <EntityTabContent value="outputs">
            <RunOutputsPanel assets={runAssets} results={outputResults} />
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
            className="m-0 flex flex-1 flex-col overflow-hidden bg-canvas p-0 text-foreground"
          >
            <RunLogsPanel
              logs={logs}
              logsError={logsError}
              selectedExecutionId={selectedExecutionId}
              attemptLabel={selectedExecutionId ? selectedExecutionId : "latest attempt"}
              onViewLatest={() => setSelectedExecutionId(null)}
            />
          </EntityTabContent>

          {runTabContributions.map((tab) => {
            const TabComponent = tab.Component;
            return (
              <EntityTabContent key={tab.id} value={tab.value}>
                {activeTab === tab.value && <TabComponent key={selectedRunId} {...props} />}
              </EntityTabContent>
            );
          })}

          {discoveredPlugins.map(({ contribution, files }) => {
            const PluginComponent = contribution.Component;
            return (
              <EntityTabContent key={contribution.id} value={contribution.value}>
                {activeTab === contribution.value && (
                  <PluginComponent key={selectedRunId} {...props} discoveredFiles={files} />
                )}
              </EntityTabContent>
            );
          })}

          <EntityTabContent value="scheduler">
            <div className="flex-1 overflow-auto p-4">
              <section>
                <h3 className="flex items-center gap-2 text-micro font-medium uppercase text-muted-foreground">
                  <Boxes className="h-3.5 w-3.5" />
                  Normalized Executor Info
                </h3>
                <div className="mt-2 overflow-hidden border-y border-border/70">
                  <Table className="w-full text-left text-body-lg">
                    <TableBody className="divide-y divide-border/50">
                      {details.map(([key, value]) => (
                        <TableRow key={key}>
                          <TableCell className="w-56 py-2 pr-4 text-label font-medium text-muted-foreground">
                            {formatExecutorLabel(key)}
                          </TableCell>
                          <TableCell className="break-all py-2 font-mono text-label text-foreground">
                            {value}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
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
