/**
 * RunActions — the run header's lifecycle verbs, gated by status so each verb
 * owns a disjoint domain (mirrors the server's run/resume/rerun split):
 *
 *   - `pending`            → Start (a dialog: pick a compute target, default
 *                            `local` = this machine, then dispatch)
 *   - `failed`/`cancelled` → Resume (continue, seeding completed tasks) · Rerun
 *                            (fresh execution from the top)
 *   - everything else      → nothing (succeeded is done; a live `running` run is
 *                            cancelled via the header's Cancel control)
 *
 * The target list always offers the built-in `local` target, so a run can be
 * started from the UI without registering anything first. Failures (e.g. a run
 * with no importable workflow entrypoint) surface inline / in an alert.
 */

import { Play, RefreshCw } from "lucide-react";
import { type JSX, useCallback, useEffect, useState } from "react";
import type { TargetResponse } from "@/api/generated/models/TargetResponse";
import { TargetsService } from "@/api/generated/services/TargetsService";
import { ExperimentsService } from "@/api/generated/services/ExperimentsService";
import { ParametersForm } from "@/app/runs/ParametersForm";
import {
  type InputField,
  parseInputSchema,
  SchemaForm,
  schemaDefaults,
} from "@/app/runs/SchemaForm";
import { workspaceApi } from "@/app/state/api";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const RETRYABLE_STATUSES = new Set(["failed", "cancelled"]);

/** Prefer the API's `detail` (FastAPI `HTTPException`) over the generic status text. */
function errMessage(err: unknown): string {
  if (err && typeof err === "object" && "body" in err) {
    const detail = (err as { body?: { detail?: unknown } }).body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) =>
          d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : String(d),
        )
        .join("; ");
    }
  }
  return err instanceof Error ? err.message : String(err);
}

export interface RunActionsProps {
  projectId: string;
  experimentId: string;
  runId: string;
  status: string;
  /** The run's current parameters — prefill for the Start dialog's inputs form. */
  params: Record<string, unknown>;
  /** Re-fetch the snapshot after a verb mutates the run. */
  onChanged: () => void;
}

export function RunActions({
  projectId,
  experimentId,
  runId,
  status,
  params,
  onChanged,
}: RunActionsProps): JSX.Element | null {
  const isPending = status === "pending";
  const isRetryable = RETRYABLE_STATUSES.has(status);

  // Start dialog
  const [startOpen, setStartOpen] = useState(false);
  const [targets, setTargets] = useState<TargetResponse[]>([]);
  const [target, setTarget] = useState("local");
  const [startParams, setStartParams] = useState<Record<string, unknown>>(params);
  const [inputSchema, setInputSchema] = useState<InputField[] | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  // Resume / rerun
  const [busy, setBusy] = useState(false);
  const [verbError, setVerbError] = useState<string | null>(null);

  useEffect(() => {
    if (!startOpen) return;
    let cancelled = false;
    TargetsService.listTargetsEndpointApiTargetsGet()
      .then((res) => {
        if (cancelled) return;
        setTargets(res.targets);
        const names = res.targets.map((t) => t.name);
        setTarget(names.includes("local") ? "local" : (names[0] ?? "local"));
      })
      .catch(() => {
        if (!cancelled) setTargets([]);
      });
    ExperimentsService.getExperimentApiProjectsProjectIdExperimentsExperimentIdGet(
      projectId,
      experimentId,
    )
      .then((exp) => {
        if (cancelled) return;
        const schema = parseInputSchema(exp.workflow);
        setInputSchema(schema);
        if (schema) setStartParams({ ...schemaDefaults(schema), ...params });
      })
      .catch(() => {
        if (!cancelled) setInputSchema(null);
      });
    return () => {
      cancelled = true;
    };
  }, [startOpen, projectId, experimentId, params]);

  const handleStart = useCallback(async (): Promise<void> => {
    setStarting(true);
    setStartError(null);
    try {
      await workspaceApi.startRun(projectId, experimentId, runId, target, startParams);
      setStartOpen(false);
      onChanged();
    } catch (err) {
      setStartError(errMessage(err));
    } finally {
      setStarting(false);
    }
  }, [projectId, experimentId, runId, target, startParams, onChanged]);

  const runVerb = useCallback(
    async (fn: () => Promise<unknown>): Promise<void> => {
      setBusy(true);
      setVerbError(null);
      try {
        await fn();
        onChanged();
      } catch (err) {
        setVerbError(errMessage(err));
      } finally {
        setBusy(false);
      }
    },
    [onChanged],
  );

  if (!isPending && !isRetryable) return null;

  return (
    <>
      {isPending && (
        <Dialog
          open={startOpen}
          onOpenChange={(open) => {
            setStartOpen(open);
            if (open) setStartParams(params);
            else setStartError(null);
          }}
        >
          <DialogTrigger asChild>
            <Button size="sm" className="h-7 gap-1">
              <Play className="h-3.5 w-3.5" />
              Start
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[480px]">
            <DialogHeader>
              <DialogTitle>Start run</DialogTitle>
              <DialogDescription>
                Set the run inputs and dispatch to a compute target.{" "}
                <code className="font-mono">local</code> runs it on this machine.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-3 py-2">
              <div className="grid gap-1.5">
                <Label>Inputs</Label>
                {inputSchema ? (
                  <SchemaForm
                    key={String(startOpen)}
                    schema={inputSchema}
                    value={startParams}
                    onChange={setStartParams}
                  />
                ) : (
                  <ParametersForm
                    key={String(startOpen)}
                    value={params}
                    onChange={setStartParams}
                  />
                )}
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="start-target">Compute target</Label>
                <Select value={target} onValueChange={setTarget}>
                  <SelectTrigger id="start-target">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {targets.map((t) => (
                      <SelectItem key={t.name} value={t.name}>
                        <span className="flex items-center gap-2">
                          <span className="font-medium">{t.name}</span>
                          <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                            {t.isRemote ? "remote" : "local"}
                          </span>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {startError && <p className="text-sm text-destructive">{startError}</p>}
            </div>
            <DialogFooter>
              <Button disabled={starting || !target} onClick={() => void handleStart()}>
                {starting ? "Starting…" : "Start run"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

      {isRetryable && (
        <>
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1"
            disabled={busy}
            title="Continue from where it stopped, seeding completed tasks"
            onClick={() =>
              void runVerb(() => workspaceApi.resumeRun(projectId, experimentId, runId))
            }
          >
            <Play className="h-3.5 w-3.5" />
            Resume
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1"
            disabled={busy}
            title="Re-execute from the top in a fresh execution"
            onClick={() =>
              void runVerb(() => workspaceApi.rerunRun(projectId, experimentId, runId))
            }
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Rerun
          </Button>
        </>
      )}

      <AlertDialog open={verbError !== null} onOpenChange={(open) => !open && setVerbError(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Action failed</AlertDialogTitle>
            <AlertDialogDescription className="break-words">{verbError}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction onClick={() => setVerbError(null)}>OK</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
