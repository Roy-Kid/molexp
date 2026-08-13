/**
 * Settings section: compute targets.
 * List/delete via TanStack Query.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, FlaskConical, Plus, Trash2, X } from "lucide-react";
import { useState } from "react";
import { TargetCreateRequest } from "@/api/generated/models/TargetCreateRequest";
import type { TargetTestResponse } from "@/api/generated/models/TargetTestResponse";
import { TargetsService } from "@/api/generated/services/TargetsService";
import { usePermissions } from "@/app/auth";
import { WorkbenchIconAction, WorkbenchTag } from "@/components/workbench";
import { AddTargetDialog } from "./AddTargetDialog";
import { settingsKeys } from "./settingsKeys";

type Scheduler = TargetCreateRequest.scheduler;

const schedulerLabel: Record<Scheduler, string> = {
  [TargetCreateRequest.scheduler.LOCAL]: "Local shell",
  [TargetCreateRequest.scheduler.SLURM]: "SLURM",
  [TargetCreateRequest.scheduler.PBS]: "PBS",
  [TargetCreateRequest.scheduler.LSF]: "LSF",
};

export function ComputeTargetsPanel(): JSX.Element {
  const { writeDeniedReason } = usePermissions();
  const queryClient = useQueryClient();
  const [busyTarget, setBusyTarget] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<TargetTestResponse | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const listQuery = useQuery({
    queryKey: settingsKeys.computeTargets(),
    queryFn: async () => {
      const res = await TargetsService.listTargetsEndpointApiTargetsGet();
      return res.targets;
    },
  });

  const targets = listQuery.data ?? [];
  const invalidate = async (): Promise<void> => {
    await queryClient.invalidateQueries({ queryKey: settingsKeys.computeTargets() });
  };

  const deleteMutation = useMutation({
    mutationFn: (name: string) => TargetsService.deleteTargetEndpointApiTargetsNameDelete(name),
    onSuccess: invalidate,
  });

  const handleDelete = async (name: string): Promise<void> => {
    setBusyTarget(name);
    setActionError(null);
    setTestResult(null);
    try {
      await deleteMutation.mutateAsync(name);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to delete target");
    } finally {
      setBusyTarget(null);
    }
  };

  const handleTest = async (name: string): Promise<void> => {
    setBusyTarget(name);
    setActionError(null);
    setTestResult(null);
    try {
      const res = await TargetsService.testTargetEndpointApiTargetsNameTestPost(name);
      setTestResult(res);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to test target");
    } finally {
      setBusyTarget(null);
    }
  };

  const listError =
    listQuery.error instanceof Error
      ? listQuery.error.message
      : listQuery.isError
        ? "Failed to list targets"
        : null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-body text-muted-foreground">
          {targets.length} {targets.length === 1 ? "target is" : "targets are"} registered.
        </p>
        {writeDeniedReason ? (
          <WorkbenchIconAction label="Add compute target" deniedReason={writeDeniedReason}>
            <Plus className="size-3.5" />
          </WorkbenchIconAction>
        ) : (
          <AddTargetDialog
            trigger={
              <WorkbenchIconAction label="Add compute target">
                <Plus className="size-3.5" />
              </WorkbenchIconAction>
            }
            onCreated={() => void invalidate()}
          />
        )}
      </div>

      <div className="space-y-3">
        {listError && <p className="text-body-lg text-status-failed-foreground">{listError}</p>}
        {listQuery.isLoading && targets.length === 0 ? (
          <p className="text-body-lg text-muted-foreground">Loading…</p>
        ) : targets.length === 0 ? (
          <p className="bg-surface/60 px-4 py-8 text-center text-body text-muted-foreground">
            No targets registered. Runs use in-process local execution until you add one.
          </p>
        ) : (
          <ul className="space-y-1">
            {targets.map((t) => (
              <li
                key={t.name}
                className="flex flex-col gap-3 bg-surface/60 px-3 py-3 transition-colors hover:bg-surface sm:flex-row sm:items-center"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate font-mono text-body font-medium">{t.name}</span>
                    <WorkbenchTag>{t.isRemote ? "ssh" : "local"}</WorkbenchTag>
                    <WorkbenchTag meaning="metadata">
                      {schedulerLabel[t.scheduler as unknown as Scheduler]}
                    </WorkbenchTag>
                  </div>
                  <div className="mt-1 truncate font-mono text-micro text-muted-foreground">
                    {t.host ? `${t.host} → ` : ""}
                    {t.scratchRoot}
                  </div>
                </div>
                <div className="flex flex-none items-center justify-end gap-1">
                  <WorkbenchIconAction
                    label={`Test ${t.name}`}
                    disabled={busyTarget === t.name}
                    deniedReason={writeDeniedReason}
                    onClick={() => void handleTest(t.name)}
                  >
                    <FlaskConical className="size-4" />
                  </WorkbenchIconAction>
                  <WorkbenchIconAction
                    label={`Remove ${t.name}`}
                    kind="ghost"
                    disabled={busyTarget === t.name}
                    deniedReason={writeDeniedReason}
                    onClick={() => void handleDelete(t.name)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </WorkbenchIconAction>
                </div>
              </li>
            ))}
          </ul>
        )}
        {actionError && <p className="text-body-lg text-status-failed-foreground">{actionError}</p>}
        {testResult && (
          <div className="space-y-1 bg-surface/70 px-3 py-3 text-body-lg">
            <div className="flex items-center gap-2 font-medium">
              {testResult.ok ? (
                <Check className="h-4 w-4 text-status-completed-foreground" />
              ) : (
                <X className="h-4 w-4 text-status-failed-foreground" />
              )}
              <span>{testResult.name}</span>
              <span className="text-muted-foreground">
                {testResult.ok ? "reachable" : "unreachable"}
              </span>
            </div>
            {testResult.error && (
              <p className="text-label text-status-failed-foreground">{testResult.error}</p>
            )}
            <ul className="space-y-1 pl-1">
              {testResult.checks.map((c) => (
                <li
                  key={c.label}
                  className="flex items-start gap-2 text-label text-muted-foreground"
                >
                  {c.ok ? (
                    <Check className="h-3 w-3 mt-1 text-status-completed-foreground flex-shrink-0" />
                  ) : (
                    <X className="h-3 w-3 mt-1 text-status-failed-foreground flex-shrink-0" />
                  )}
                  <span>
                    {c.label}
                    {c.detail && (
                      <span className="text-status-failed-foreground"> — {c.detail}</span>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
