/**
 * useRunViewer — the shared state core behind both run viewers.
 *
 * `RunViewer` (the default run renderer) and `MolqRunViewer` (the molq-backend
 * override) draw very different chrome but drive it from identical state: run
 * resolution off the snapshot, the active-tab sync, lazy stdout/stderr fetching
 * per execution, and the "mark cancelled" + copy-id handlers. That logic lives
 * here once; each component keeps only its own layout. Run-asset counts and the
 * file-type discovery tabs are RunViewer-only and stay in that component.
 *
 * All hooks run unconditionally (run may be null — effects guard on its id), so
 * callers can still early-return their own "run not found" chrome afterwards.
 */

import { type ReactNode, useEffect, useMemo, useState } from "react";
import { listEntityTabs } from "@/app/registry";
import { formatDuration } from "@/app/renderers/dashboardData";
import { workspaceApi } from "@/app/state/api";
import { useInspectedTask } from "@/app/state/inspectedTask";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps, WorkspaceSnapshot } from "@/app/types";
import { useAlert, useConfirm } from "@/components/ConfirmDialog";

type RunRow = WorkspaceSnapshot["runs"][number];
type RunLogs = { stdout?: string | null; stderr?: string | null } | null;

const terminalRunStatuses = new Set(["succeeded", "failed", "cancelled", "skipped"]);

export interface UseRunViewer {
  run: RunRow | null;
  project: WorkspaceSnapshot["projects"][number] | undefined;
  experiment: WorkspaceSnapshot["experiments"][number] | undefined;
  workflow: WorkspaceSnapshot["workflows"][number] | undefined;
  selectedRunId: string;
  activeTab: string;
  setActiveTab: (tab: string) => void;
  logs: RunLogs;
  logsError: string | null;
  selectedExecutionId: string | null;
  setSelectedExecutionId: (id: string | null) => void;
  duration: string | null;
  attemptCount: number;
  parameterEntries: [string, unknown][];
  resultEntries: [string, unknown][];
  isTerminal: boolean;
  runTabContributions: ReturnType<typeof listEntityTabs>;
  inspectTask: ReturnType<typeof useInspectedTask>["inspectTask"];
  setSelection: ReturnType<typeof useNavigationState>["setSelection"];
  handleCopyRunId: () => void;
  handleCancelRun: () => Promise<void>;
  confirmDialog: ReactNode;
  alertDialog: ReactNode;
}

export const useRunViewer = (props: RendererProps): UseRunViewer => {
  const { selection, snapshot, onRefresh } = props;
  const { setSelection } = useNavigationState(snapshot);
  const { inspectTask } = useInspectedTask();
  const [logs, setLogs] = useState<RunLogs>(null);
  const [logsError, setLogsError] = useState<string | null>(null);
  const [selectedExecutionId, setSelectedExecutionId] = useState<string | null>(null);
  const runTabContributions = listEntityTabs("run");
  const { confirm, dialog: confirmDialog } = useConfirm();
  const { alert, dialog: alertDialog } = useAlert();

  const run = useMemo(
    () => snapshot.runs.find((item) => item.id === selection.objectId) ?? null,
    [snapshot.runs, selection.objectId],
  );

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

  const project = run ? snapshot.projects.find((item) => item.id === run.projectId) : undefined;
  const experiment = run
    ? snapshot.experiments.find((item) => item.id === run.experimentId)
    : undefined;
  const workflow =
    run && experiment
      ? snapshot.workflows.find(
          (item) =>
            item.experimentId === experiment.id &&
            (item.name === experiment.workflowFile || item.id === experiment.workflowFile),
        )
      : undefined;

  const duration = run ? formatDuration(run.startedAt, run.finishedAt) : null;
  const attemptCount = run?.executionHistory.length ?? 0;
  const parameterEntries = Object.entries(run?.parameters ?? {});
  const resultEntries = Object.entries(run?.results ?? {});
  const isTerminal = run ? terminalRunStatuses.has(run.status) : true;

  const handleCopyRunId = (): void => {
    if (!run) return;
    void navigator.clipboard.writeText(run.id);
  };

  const handleCancelRun = async (): Promise<void> => {
    if (!run || terminalRunStatuses.has(run.status)) return;
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

  return {
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
  };
};
