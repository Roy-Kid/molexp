/**
 * Settings section for managing the workspace's registered ComputeTargets
 * — the cross-product of transport × scheduler that runs can be submitted to.
 */

import { Check, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { TargetCreateRequest } from "@/api/generated/models/TargetCreateRequest";
import type { TargetResponse } from "@/api/generated/models/TargetResponse";
import type { TargetTestResponse } from "@/api/generated/models/TargetTestResponse";
import { TargetsService } from "@/api/generated/services/TargetsService";
import { WorkbenchAction, WorkbenchIconAction, WorkbenchTag } from "@/components/workbench";
import { AddTargetForm } from "./AddTargetForm";

type Scheduler = TargetCreateRequest.scheduler;

const schedulerLabel: Record<Scheduler, string> = {
  [TargetCreateRequest.scheduler.LOCAL]: "Local shell",
  [TargetCreateRequest.scheduler.SLURM]: "SLURM",
  [TargetCreateRequest.scheduler.PBS]: "PBS",
  [TargetCreateRequest.scheduler.LSF]: "LSF",
};

export function ComputeTargetsPanel(): JSX.Element {
  const [targets, setTargets] = useState<TargetResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  const [busyTarget, setBusyTarget] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<TargetTestResponse | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setListError(null);
    try {
      const res = await TargetsService.listTargetsEndpointApiTargetsGet();
      setTargets(res.targets);
    } catch (err) {
      setListError(err instanceof Error ? err.message : "Failed to list targets");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const handleDelete = async (name: string) => {
    setBusyTarget(name);
    setActionError(null);
    setTestResult(null);
    try {
      await TargetsService.deleteTargetEndpointApiTargetsNameDelete(name);
      await refresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to delete target");
    } finally {
      setBusyTarget(null);
    }
  };

  const handleTest = async (name: string) => {
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

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      {/* List + test results */}
      <section className="space-y-3">
        <header>
          <h3 className="sr-only">Compute targets</h3>
          <p className="text-xs text-muted-foreground">
            {targets.length} registered · Runs dispatch through a local shell, SSH host, or batch
            scheduler.
          </p>
        </header>
        {listError && <p className="text-sm text-status-failed-foreground">{listError}</p>}
        {loading && targets.length === 0 ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : targets.length === 0 ? (
          <p className="border-y border-dashed border-border/70 py-6 text-center text-sm text-muted-foreground">
            No targets registered. Add one on the right — runs default to in-process local execution
            until a target is selected.
          </p>
        ) : (
          <ul className="divide-y divide-border border-y border-border">
            {targets.map((t) => (
              <li key={t.name} className="flex items-center gap-3 px-3 py-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-medium truncate">{t.name}</span>
                    <WorkbenchTag>{t.isRemote ? "ssh" : "local"}</WorkbenchTag>
                    <WorkbenchTag meaning="metadata">
                      {schedulerLabel[t.scheduler as unknown as Scheduler]}
                    </WorkbenchTag>
                  </div>
                  <div className="text-xs text-muted-foreground truncate">
                    {t.host ? `${t.host} → ` : ""}
                    {t.scratchRoot}
                  </div>
                </div>
                <WorkbenchAction
                  kind="ghost"
                  size="compact"
                  disabled={busyTarget === t.name}
                  onClick={() => handleTest(t.name)}
                >
                  Test
                </WorkbenchAction>
                <WorkbenchIconAction
                  label={`Remove ${t.name}`}
                  kind="ghost"
                  disabled={busyTarget === t.name}
                  onClick={() => handleDelete(t.name)}
                >
                  <Trash2 className="h-4 w-4" />
                </WorkbenchIconAction>
              </li>
            ))}
          </ul>
        )}
        {actionError && <p className="text-sm text-status-failed-foreground">{actionError}</p>}
        {testResult && (
          <div className="space-y-1 border-y border-border/60 bg-muted/30 px-3 py-3 text-sm">
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
              <p className="text-xs text-status-failed-foreground">{testResult.error}</p>
            )}
            <ul className="space-y-1 pl-1">
              {testResult.checks.map((c) => (
                <li key={c.label} className="flex items-start gap-2 text-xs text-muted-foreground">
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
      </section>

      {/* Add form */}
      <AddTargetForm onCreated={() => void refresh()} />
    </div>
  );
}
