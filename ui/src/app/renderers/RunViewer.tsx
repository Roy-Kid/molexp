import { FileQuestion, PlayCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  DashboardCard,
  DashboardGrid,
  EmptyState,
  EntityPage,
  EntityPath,
  MetaField,
  MetaGrid,
  StatCard,
  StatGrid,
} from "@/app/components/entity";
// Module-path import (not the barrel) — see KnowledgeBacklinksCard loader note.
import { KnowledgeBacklinksCard } from "@/app/components/entity/KnowledgeBacklinksCard";
import { formatScalar } from "@/app/renderers/dashboardData";
import { RunExecutionsPanel } from "@/app/renderers/RunExecutionsPanel";
import { RunLogsPanel } from "@/app/renderers/RunLogsPanel";
import { useRunViewer } from "@/app/renderers/useRunViewer";
import { RunMetricsView } from "@/app/runs/metrics/RunMetricsView";
import { POST_DISPATCH_TAB, RunToolbar } from "@/app/runs/RunToolbar";
import { workspaceApi } from "@/app/state/api";
import { useDiscoveredFileTypesForRun } from "@/app/state/useDiscoveredFileTypes";
import type { ApiAssetResponse, RendererProps } from "@/app/types";
import { formatDateTime } from "@/lib/datetime";

const openKnowledgePath = (
  path: string,
  setSelection: (sel: { objectType: "knowledge"; objectId: string }) => void,
): void => {
  if (!path) {
    setSelection({ objectType: "knowledge", objectId: "" });
    return;
  }
  const rel = path.startsWith("/") ? path.split("/").filter(Boolean).slice(-3).join("/") : path;
  setSelection({ objectType: "knowledge", objectId: rel });
};

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
        <EmptyState icon={<FileQuestion className="h-5 w-5" />} title="Run not found" />
      </div>
    );
  }

  const backend = run.executorInfo.backend || "local";

  const overviewContent = (
    <DashboardGrid>
      {run.errorMessage && (
        <DashboardCard title="Error" variant="destructive" className="lg:col-span-12">
          <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-destructive">
            {run.errorMessage}
          </pre>
        </DashboardCard>
      )}

      <div className="lg:col-span-12">
        <StatGrid>
          <StatCard label="Duration" value={duration ?? "—"} muted={!duration} />
          <StatCard
            label="Attempts"
            value={attemptCount || 0}
            muted={!attemptCount}
          />
          <StatCard label="Backend" value={backend} />
          <StatCard
            label="Assets"
            value={runAssets.length}
            muted={runAssets.length === 0}
          />
          <StatCard
            label="Results"
            value={resultEntries.length}
            muted={resultEntries.length === 0}
          />
        </StatGrid>
      </div>

      <DashboardCard
        title="Summary"
        description="Timing and placement"
        className="lg:col-span-8"
        bodyClassName="space-y-4"
      >
        <MetaGrid columns={3}>
          <MetaField
            label="Started"
            value={formatDateTime(run.startedAt)}
            title={run.startedAt ?? undefined}
          />
          <MetaField
            label="Finished"
            value={formatDateTime(run.finishedAt)}
            title={run.finishedAt ?? undefined}
          />
          <MetaField label="Backend" value={backend} mono />
        </MetaGrid>
        {run.summary && (
          <p className="text-sm leading-relaxed text-muted-foreground">{run.summary}</p>
        )}
        <EntityPath
          segments={[
            {
              label: project?.name ?? run.projectId,
              onClick: () => setSelection({ objectType: "project", objectId: run.projectId }),
            },
            {
              label: experiment?.name ?? run.experimentId,
              onClick: () =>
                setSelection({ objectType: "experiment", objectId: run.experimentId }),
            },
            ...(workflow
              ? [
                  {
                    label: workflow.name || "workflow",
                    onClick: () =>
                      setSelection({
                        objectType: "workflow" as const,
                        objectId: workflow.id,
                        workflowId: workflow.id,
                      }),
                  },
                ]
              : []),
          ]}
          trailing={run.id}
        />
      </DashboardCard>

      <KnowledgeBacklinksCard
        kind="run"
        projectId={run.projectId}
        experimentId={run.experimentId}
        runId={run.id}
        className="lg:col-span-4"
      />

      <DashboardCard
        title="Parameters"
        description={
          parameterEntries.length === 0
            ? "None set"
            : `${parameterEntries.length} parameter${parameterEntries.length === 1 ? "" : "s"}`
        }
        className="lg:col-span-6"
      >
        {parameterEntries.length === 0 ? (
          <p className="text-sm text-muted-foreground">No parameters on this run.</p>
        ) : (
          <MetaGrid columns={2}>
            {parameterEntries.map(([key, value]) => (
              <MetaField
                key={key}
                label={key}
                value={formatScalar(value)}
                mono
                title={formatScalar(value)}
              />
            ))}
          </MetaGrid>
        )}
      </DashboardCard>

      <DashboardCard
        title="Results"
        description={
          resultEntries.length === 0
            ? "Nothing recorded"
            : `${resultEntries.length} result field${resultEntries.length === 1 ? "" : "s"}`
        }
        className="lg:col-span-6"
      >
        {resultEntries.length === 0 ? (
          <p className="text-sm text-muted-foreground">No results recorded yet.</p>
        ) : (
          <MetaGrid columns={2}>
            {resultEntries.map(([key, value]) => (
              <MetaField
                key={key}
                label={key}
                value={formatScalar(value)}
                mono
                title={formatScalar(value)}
              />
            ))}
          </MetaGrid>
        )}
      </DashboardCard>
    </DashboardGrid>
  );

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
    attemptLabel = `#${selectedExecutionIndex + 1}`;
  } else if (attemptCountForLabel > 0) {
    attemptLabel = `#${attemptCountForLabel}`;
  } else {
    attemptLabel = "latest";
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

  const hasLogs = Boolean(logs?.stdout || logs?.stderr);
  const tabs = [
    { value: "overview", label: "Overview", content: overviewContent },
    {
      value: "executions",
      label: attemptCount ? `Executions (${attemptCount})` : "Executions",
      content: executionsContent,
    },
    ...(hasLogs ? [{ value: "logs", label: "Logs", content: logsContent }] : []),
    {
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
              if (path) openKnowledgePath(path, setSelection);
            }}
          />
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
