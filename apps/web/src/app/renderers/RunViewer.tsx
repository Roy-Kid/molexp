import { FileQuestion, PlayCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { EmptyState, EntityPage } from "@/app/components/entity";
import { RunExecutionsPanel } from "@/app/renderers/RunExecutionsPanel";
import { RunLogsPanel } from "@/app/renderers/RunLogsPanel";
import { RunOutputsPanel } from "@/app/renderers/run/RunOutputsPanel";
import { RunOverview } from "@/app/renderers/run/RunOverview";
import { useRunViewer } from "@/app/renderers/useRunViewer";
import { POST_DISPATCH_TAB, RunToolbar } from "@/app/runs/RunToolbar";
import { workspaceApi } from "@/app/state/api";
import { useDiscoveredFileTypesForRun } from "@/app/state/useDiscoveredFileTypes";
import type { ApiAssetResponse, RendererProps } from "@/app/types";

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
    <RunOverview
      run={run}
      backend={backend}
      duration={duration}
      attemptCount={attemptCount}
      assets={runAssets}
      parameters={parameterEntries}
      results={resultEntries}
    />
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
  const outputResults = resultEntries.map(([key, value]) => ({ key, value }));
  const outputsContent = <RunOutputsPanel assets={runAssets} results={outputResults} />;
  const tabs = [
    { value: "overview", label: "Overview", content: overviewContent },
    {
      value: "outputs",
      label:
        runAssets.length + resultEntries.length > 0
          ? `Outputs (${runAssets.length + resultEntries.length})`
          : "Outputs",
      content: outputsContent,
    },
    {
      value: "executions",
      label: attemptCount ? `Executions (${attemptCount})` : "Executions",
      content: executionsContent,
    },
    ...(hasLogs ? [{ value: "logs", label: "Logs", content: logsContent }] : []),
    // Domain tabs (molvis, metrics plugin if *.mlp.jsonl present, …) — data-driven only.
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
