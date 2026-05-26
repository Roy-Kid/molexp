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
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

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
        <header className="flex items-baseline justify-between">
          <h3 className="text-sm font-semibold text-foreground">
            Compute targets <span className="text-muted-foreground">({targets.length})</span>
          </h3>
          <p className="text-xs text-muted-foreground">
            Where runs are dispatched: local shell, SSH host, or batch scheduler
          </p>
        </header>
        {listError && <p className="text-sm text-red-500">{listError}</p>}
        {loading && targets.length === 0 ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : targets.length === 0 ? (
          <p className="rounded-md border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
            No targets registered. Add one on the right — runs default to in-process local execution
            until a target is selected.
          </p>
        ) : (
          <ul className="divide-y divide-border rounded-md border border-border">
            {targets.map((t) => (
              <li key={t.name} className="flex items-center gap-3 px-3 py-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-medium truncate">{t.name}</span>
                    <Badge variant={t.isRemote ? "default" : "secondary"}>
                      {t.isRemote ? "ssh" : "local"}
                    </Badge>
                    <Badge variant="outline">
                      {schedulerLabel[t.scheduler as unknown as Scheduler]}
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground truncate">
                    {t.host ? `${t.host} → ` : ""}
                    {t.scratchRoot}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={busyTarget === t.name}
                  onClick={() => handleTest(t.name)}
                >
                  Test
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={`Remove ${t.name}`}
                  disabled={busyTarget === t.name}
                  onClick={() => handleDelete(t.name)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </li>
            ))}
          </ul>
        )}
        {actionError && <p className="text-sm text-red-500">{actionError}</p>}
        {testResult && (
          <div className="rounded-md border border-border bg-muted/30 p-3 text-sm space-y-1">
            <div className="flex items-center gap-2 font-medium">
              {testResult.ok ? (
                <Check className="h-4 w-4 text-green-500" />
              ) : (
                <X className="h-4 w-4 text-red-500" />
              )}
              <span>{testResult.name}</span>
              <span className="text-muted-foreground">
                {testResult.ok ? "reachable" : "unreachable"}
              </span>
            </div>
            {testResult.error && <p className="text-xs text-red-500">{testResult.error}</p>}
            <ul className="space-y-0.5 pl-1">
              {testResult.checks.map((c) => (
                <li
                  key={c.label}
                  className="flex items-start gap-1.5 text-xs text-muted-foreground"
                >
                  {c.ok ? (
                    <Check className="h-3 w-3 mt-0.5 text-green-500 flex-shrink-0" />
                  ) : (
                    <X className="h-3 w-3 mt-0.5 text-red-500 flex-shrink-0" />
                  )}
                  <span>
                    {c.label}
                    {c.detail && <span className="text-red-500"> — {c.detail}</span>}
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
