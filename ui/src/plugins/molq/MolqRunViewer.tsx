import { Ban, Boxes, Copy, FileQuestion, ServerCog, Terminal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  EmptyState,
  EntityHeader,
  EntityMetric,
  EntityTabBar,
  EntityTabContent,
  EntityTabs,
  KeyValueGrid,
  OverviewHighlight,
  OverviewHighlightGrid,
  OverviewPage,
  OverviewSection,
} from "@/app/components/entity";
import { listEntityTabs } from "@/app/registry";
import { RunViewer } from "@/app/renderers/RunViewer";
import { RunSnapshotPanel } from "@/app/renderers/SnapshotViewer";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";
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
  const { setSelection, breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);
  const [logs, setLogs] = useState<{ stdout?: string | null; stderr?: string | null } | null>(null);
  const [logsError, setLogsError] = useState<string | null>(null);
  const runTabContributions = listEntityTabs("run");

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

  useEffect(() => {
    let cancelled = false;
    setLogsError(null);

    if (!run || activeTab !== "logs") {
      return;
    }

    setLogs(null);
    workspaceApi
      .getRunLogs(run.projectId, run.experimentId, run.id)
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
  }, [activeTab, run]);

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
  const scheduler = getExecutorEntry(run.executorInfo, "scheduler") ?? "unknown";
  const cluster = getExecutorEntry(run.executorInfo, "cluster_name", "cluster") ?? "default";
  const jobId = getExecutorEntry(run.executorInfo, "job_id") ?? "pending";
  const schedulerJobId = getExecutorEntry(run.executorInfo, "scheduler_job_id") ?? "not assigned";
  const details = Object.entries(run.executorInfo);
  const executorFields = [
    { label: "Backend", value: <span className="font-mono text-xs">molq</span> },
    { label: "Scheduler", value: <span className="font-mono text-xs">{scheduler}</span> },
    { label: "Cluster", value: <span className="font-mono text-xs">{cluster}</span> },
    { label: "Job ID", value: <span className="font-mono text-xs">{jobId}</span> },
    {
      label: "Scheduler Job ID",
      value: <span className="font-mono text-xs">{schedulerJobId}</span>,
    },
  ];

  const handleCopyRunId = (): void => {
    void navigator.clipboard.writeText(run.id);
  };

  const handleTabChange = (value: string): void => {
    setActiveTab(value);
  };

  const handleCancelRun = async (): Promise<void> => {
    if (terminalRunStatuses.has(run.status)) return;
    const confirmed = window.confirm(
      `Mark run "${run.id}" as cancelled?\n\nThis updates workspace status only; it does not cancel a scheduler job.`,
    );
    if (!confirmed) return;

    try {
      await workspaceApi.updateRunStatus(run.projectId, run.experimentId, run.id, "cancelled");
      onRefresh();
    } catch (error) {
      console.error("Failed to mark run cancelled:", error);
      window.alert("Failed to mark run cancelled");
    }
  };

  return (
    <div className="flex h-full flex-col bg-background">
      <EntityHeader
        breadcrumbs={breadcrumbs}
        canNavigateUp={canNavigateUp}
        onNavigateUp={navigateUp}
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
              { value: "logs", label: "Logs" },
              ...runTabContributions.map((tab) => ({ value: tab.value, label: tab.label })),
              { value: "scheduler", label: "Scheduler" },
              { value: "snapshot", label: "Snapshot" },
            ]}
          />

          <EntityTabContent value="overview">
            <OverviewPage
              aside={
                <>
                  <OverviewSection title="Highlights">
                    <OverviewHighlightGrid>
                      <OverviewHighlight label="Scheduler" value={scheduler} />
                      <OverviewHighlight label="Cluster" value={cluster} />
                      <OverviewHighlight label="Status" value={run.status} />
                      <OverviewHighlight label="Job ID" value={jobId} />
                    </OverviewHighlightGrid>
                  </OverviewSection>

                  <OverviewSection title="Relationships">
                    <div className="flex flex-wrap gap-1.5">
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() =>
                          setSelection({ objectType: "project", objectId: run.projectId })
                        }
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
                    </div>
                  </OverviewSection>
                </>
              }
            >
              <OverviewSection title="Summary">
                <p className="max-w-3xl text-sm leading-6 text-foreground">
                  {run.summary || (
                    <span className="text-muted-foreground">
                      No summary provided for this molq-backed run.
                    </span>
                  )}
                </p>
              </OverviewSection>

              <OverviewSection title="Execution">
                <KeyValueGrid items={executorFields} />
              </OverviewSection>

              <OverviewSection title="Scheduler Metadata">
                <KeyValueGrid
                  items={details.map(([key, value]) => ({
                    label: formatExecutorLabel(key),
                    value: <span className="font-mono text-xs">{value}</span>,
                  }))}
                />
              </OverviewSection>
            </OverviewPage>
          </EntityTabContent>

          <EntityTabContent
            value="logs"
            className="m-0 flex flex-1 flex-col overflow-hidden bg-zinc-950 p-0 text-zinc-50 dark:bg-black"
          >
            <div className="flex items-center gap-2 border-b border-zinc-800 bg-zinc-900 px-3 py-1 font-mono text-[11px] text-zinc-400">
              <Terminal className="h-3 w-3" />
              stdout/stderr
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

          <EntityTabContent value="snapshot">
            <RunSnapshotPanel runId={run.id} />
          </EntityTabContent>
        </EntityTabs>
      </div>
    </div>
  );
};
