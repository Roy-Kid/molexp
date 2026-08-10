import { FileQuestion, ServerCog } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  DashboardCanvas,
  EmptyState,
  EntityHeader,
  EntityTabBar,
  EntityTabContent,
  EntityTabs,
  OverviewSurface,
} from "@/app/components/entity";
import { formatScalar } from "@/app/renderers/dashboardData";
import { RunExecutionsPanel } from "@/app/renderers/RunExecutionsPanel";
import { RunLogsPanel } from "@/app/renderers/RunLogsPanel";
import { RunViewer } from "@/app/renderers/RunViewer";
import { RunOutputsPanel } from "@/app/renderers/run/RunOutputsPanel";
import { useRunViewer } from "@/app/renderers/useRunViewer";
import { POST_DISPATCH_TAB, RunToolbar } from "@/app/runs/RunToolbar";
import { workspaceApi } from "@/app/state/api";
import { useDiscoveredFileTypesForRun } from "@/app/state/useDiscoveredFileTypes";
import type { ApiAssetResponse, RendererProps } from "@/app/types";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatDateTime } from "@/lib/datetime";

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

export const MolqRunViewer = (props: RendererProps): JSX.Element => {
  const {
    run,
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
  const outputResults = resultEntries.map(([key, value]) => ({ key, value }));

  const keyValueTable = (entries: [string, unknown][], emptyLabel: string): JSX.Element =>
    entries.length === 0 ? (
      <p className="py-4 text-label text-muted-foreground">{emptyLabel}</p>
    ) : (
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-2/5">Key</TableHead>
            <TableHead>Value</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {entries.map(([key, value]) => {
            const text = formatScalar(value);
            return (
              <TableRow key={key}>
                <TableCell className="align-top text-label text-muted-foreground">{key}</TableCell>
                <TableCell className="break-all font-mono text-label text-foreground" title={text}>
                  {text}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    );

  const ops: string[] = [];
  if (run.startedAt) ops.push(`started ${formatDateTime(run.startedAt)}`);
  if (duration) ops.push(duration);
  ops.push(scheduler, cluster);
  if (jobId !== "pending") ops.push(`job ${jobId}`);
  if (attemptCount > 1) ops.push(`${attemptCount} attempts`);
  if (runAssets.length > 0) ops.push(`${runAssets.length} assets`);

  return (
    <div className="flex h-full flex-col bg-background">
      <EntityHeader
        icon={ServerCog}
        title={run.name}
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
              // Plugin tabs (molq, metrics, …) — data-driven only.
              ...runTabContributions.map((tab) => ({ value: tab.value, label: tab.label })),
              ...discoveredPlugins.map(({ contribution, files }) => ({
                value: contribution.value,
                label: `${contribution.label} (${files.length})`,
              })),
            ]}
          />

          <EntityTabContent value="overview">
            <OverviewSurface>
              <DashboardCanvas className="max-w-4xl space-y-10">
                {run.errorMessage && (
                  <section
                    className="rounded-panel border border-status-failed/25 bg-status-failed-soft px-4 py-3"
                    aria-label="Run error"
                  >
                    <p className="text-label font-medium text-status-failed-foreground">Error</p>
                    <pre className="mt-1.5 whitespace-pre-wrap break-words font-mono text-label leading-relaxed text-status-failed-foreground">
                      {run.errorMessage}
                    </pre>
                  </section>
                )}

                {ops.length > 0 && (
                  <p className="font-mono text-micro tabular-nums text-muted-foreground">
                    {ops.join(" · ")}
                  </p>
                )}

                {run.summary ? (
                  <p className="max-w-2xl text-body leading-relaxed text-muted-foreground">
                    {run.summary}
                  </p>
                ) : null}

                <div className="grid gap-10 lg:grid-cols-2">
                  <section className="min-w-0 space-y-3">
                    <h3 className="text-body-lg font-medium text-foreground">
                      Parameters
                      <span className="ml-2 font-mono text-micro font-normal text-muted-foreground">
                        {parameterEntries.length}
                      </span>
                    </h3>
                    {keyValueTable(parameterEntries, "No parameters")}
                  </section>
                  <section className="min-w-0 space-y-3">
                    <h3 className="text-body-lg font-medium text-foreground">
                      Results
                      <span className="ml-2 font-mono text-micro font-normal text-muted-foreground">
                        {resultEntries.length}
                      </span>
                    </h3>
                    {keyValueTable(resultEntries, "No results")}
                  </section>
                </div>
              </DashboardCanvas>
            </OverviewSurface>
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
        </EntityTabs>
      </div>
      {confirmDialog}
      {alertDialog}
    </div>
  );
};
