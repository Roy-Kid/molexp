import { Ban, Copy, FileQuestion, FileText, Terminal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  EmptyState,
  EntityHeader,
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
import { buildMetadataFields } from "@/app/renderers/metadata";
import { RunSnapshotPanel } from "@/app/renderers/SnapshotViewer";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";
import { Button } from "@/components/ui/button";

const terminalRunStatuses = new Set(["succeeded", "failed", "cancelled", "skipped"]);

export const RunViewer = (props: RendererProps): JSX.Element => {
  const { selection, snapshot, onRefresh } = props;
  const { setSelection, breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);
  const fields = buildMetadataFields(selection, snapshot);
  const [logs, setLogs] = useState<{ stdout?: string | null; stderr?: string | null } | null>(null);
  const [logsError, setLogsError] = useState<string | null>(null);
  const runTabContributions = listEntityTabs("run");

  const run = useMemo(() => {
    return snapshot.runs.find((r) => r.id === selection.objectId);
  }, [snapshot.runs, selection.objectId]);

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

  const project = snapshot.projects.find((item) => item.id === run.projectId);
  const experiment = snapshot.experiments.find((item) => item.id === run.experimentId);

  // Filter interesting fields to show in the table
  const displayFields = fields.filter(
    (f) => !["Run", "Status", "Summary", "Project", "Experiment"].includes(f.label),
  );

  const handleCopyRunId = (): void => {
    void navigator.clipboard.writeText(run.id);
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
      />

      <div className="flex-1 overflow-hidden flex flex-col">
        <EntityTabs value={activeTab} onValueChange={setActiveTab}>
          <EntityTabBar
            tabs={[
              { value: "overview", label: "Overview" },
              { value: "logs", label: "Logs" },
              ...runTabContributions.map((tab) => ({ value: tab.value, label: tab.label })),
              { value: "snapshot", label: "Snapshot" },
            ]}
          />

          <EntityTabContent value="overview">
            <OverviewPage
              aside={
                <>
                  <OverviewSection title="Highlights">
                    <OverviewHighlightGrid>
                      <OverviewHighlight label="Status" value={run.status} />
                      <OverviewHighlight
                        label="Updated"
                        value={new Date(run.updatedAt).toLocaleString()}
                      />
                      <OverviewHighlight
                        label="Backend"
                        value={run.executorInfo.backend || "local"}
                      />
                      <OverviewHighlight label="Run ID" value={run.id} />
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
                    <span className="text-muted-foreground">No summary provided.</span>
                  )}
                </p>
              </OverviewSection>

              <OverviewSection title="Metadata">
                <KeyValueGrid
                  items={displayFields.map((field) => ({
                    label: field.label,
                    value: <span className="font-mono text-xs">{field.value}</span>,
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

          <EntityTabContent value="snapshot">
            <RunSnapshotPanel runId={run.id} />
          </EntityTabContent>

          {runTabContributions.map((tab) => {
            const TabComponent = tab.Component;
            return (
              <EntityTabContent key={tab.id} value={tab.value}>
                {activeTab === tab.value && <TabComponent key={selectedRunId} {...props} />}
              </EntityTabContent>
            );
          })}
        </EntityTabs>
      </div>
    </div>
  );
};
