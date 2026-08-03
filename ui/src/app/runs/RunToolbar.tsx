/**
 * Run header toolbar — one place for every status-gated action.
 *
 * Layout (left → right):
 *   1. Lifecycle (Start | Resume/Rerun/Fresh | Cancel) — only what status allows
 *   2. Harvest — when outcome can become knowledge
 *   3. More menu — Export / Copy / Agent (always available)
 *
 * Shared by RunViewer and MolqRunViewer so molq runs get the same verbs.
 */

import { Ban, Bot, Copy, Download, MoreHorizontal, Play, RefreshCw } from "lucide-react";
import { type JSX, useCallback, useEffect, useState } from "react";
import type { TargetResponse } from "@/api/generated/models/TargetResponse";
import { ExperimentsService } from "@/api/generated/services/ExperimentsService";
import { TargetsService } from "@/api/generated/services/TargetsService";
import { HarvestDialog } from "@/app/components/HarvestDialog";
import { ParametersForm } from "@/app/runs/ParametersForm";
import {
  canCancel,
  canHarvest,
  canRerun,
  canResume,
  canStart,
  POST_DISPATCH_TAB,
} from "@/app/runs/runLifecycle";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "@/components/ui/toast";
import { WorkbenchAction, WorkbenchIconAction } from "@/components/workbench";

function errMessage(err: unknown): string {
  if (err && typeof err === "object" && "body" in err) {
    const detail = (err as { body?: { detail?: unknown } }).body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) =>
          d && typeof d === "object" && "msg" in d
            ? String((d as { msg: unknown }).msg)
            : String(d),
        )
        .join("; ");
    }
  }
  return err instanceof Error ? err.message : String(err);
}

export interface RunToolbarProps {
  projectId: string;
  experimentId: string;
  runId: string;
  status: string;
  params: Record<string, unknown>;
  onRefresh: () => void;
  onCancel: () => Promise<void>;
  /** After start / resume / rerun — typically open Executions. */
  onDispatched?: () => void;
  onOpenAgent: () => void;
  onHarvested: (path: string) => void;
}

export function RunToolbar({
  projectId,
  experimentId,
  runId,
  status,
  params,
  onRefresh,
  onCancel,
  onDispatched,
  onOpenAgent,
  onHarvested,
}: RunToolbarProps): JSX.Element {
  const showStart = canStart(status);
  const showCancel = canCancel(status);
  const showRetry = canResume(status) && canRerun(status);
  const showHarvest = canHarvest(status);

  const [startOpen, setStartOpen] = useState(false);
  const [targets, setTargets] = useState<TargetResponse[]>([]);
  const [target, setTarget] = useState("local");
  const [startParams, setStartParams] = useState<Record<string, unknown>>(params);
  const [inputSchema, setInputSchema] = useState<InputField[] | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
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

  const afterDispatch = useCallback((): void => {
    onRefresh();
    onDispatched?.();
  }, [onRefresh, onDispatched]);

  const handleStart = useCallback(async (): Promise<void> => {
    setStarting(true);
    setStartError(null);
    try {
      await workspaceApi.startRun(projectId, experimentId, runId, target, startParams);
      setStartOpen(false);
      toast.success("Started");
      afterDispatch();
    } catch (err) {
      setStartError(errMessage(err));
    } finally {
      setStarting(false);
    }
  }, [projectId, experimentId, runId, target, startParams, afterDispatch]);

  const runVerb = useCallback(
    async (label: string, fn: () => Promise<unknown>): Promise<void> => {
      setBusy(true);
      setVerbError(null);
      try {
        await fn();
        toast.success(label);
        afterDispatch();
      } catch (err) {
        setVerbError(errMessage(err));
      } finally {
        setBusy(false);
      }
    },
    [afterDispatch],
  );

  const exportUrl = workspaceApi.runExportUrl(projectId, experimentId, runId);

  return (
    <>
      <div className="flex items-center gap-1">
        {/* ── Lifecycle (status-disjoint) ─────────────────────────────── */}
        {showStart && (
          <Dialog
            open={startOpen}
            onOpenChange={(open) => {
              setStartOpen(open);
              if (open) setStartParams(params);
              else setStartError(null);
            }}
          >
            <DialogTrigger asChild>
              <WorkbenchAction kind="primary" size="compact" className="h-7 gap-1">
                <Play className="h-3.5 w-3.5" />
                Start
              </WorkbenchAction>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[480px]">
              <DialogHeader>
                <DialogTitle>Start</DialogTitle>
                <DialogDescription className="sr-only">Inputs and target</DialogDescription>
              </DialogHeader>
              <div className="grid gap-3 py-2">
                <div className="grid gap-2">
                  <span className="text-label font-medium">Inputs</span>
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
                <div className="grid gap-2">
                  <Label htmlFor="start-target">Target</Label>
                  <Select value={target} onValueChange={setTarget}>
                    <SelectTrigger id="start-target">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {targets.map((t) => (
                        <SelectItem key={t.name} value={t.name}>
                          <span className="flex items-center gap-2">
                            <span className="font-medium">{t.name}</span>
                            <span className="text-micro uppercase text-muted-foreground">
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
                <WorkbenchAction
                  kind="primary"
                  size="default"
                  disabled={starting || !target}
                  onClick={() => void handleStart()}
                >
                  {starting ? "…" : "Start"}
                </WorkbenchAction>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}

        {showRetry && (
          <>
            <WorkbenchAction
              kind="primary"
              size="compact"
              className="h-7 gap-1"
              disabled={busy}
              title="Continue last execution"
              onClick={() =>
                void runVerb("Resumed", () =>
                  workspaceApi.resumeRun(projectId, experimentId, runId),
                )
              }
            >
              <Play className="h-3.5 w-3.5" />
              Resume
            </WorkbenchAction>
            <WorkbenchAction
              kind="secondary"
              size="compact"
              className="h-7 gap-1"
              disabled={busy}
              title="New execution from top"
              onClick={() =>
                void runVerb("Rerun", () =>
                  workspaceApi.rerunRun(projectId, experimentId, runId, false),
                )
              }
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Rerun
            </WorkbenchAction>
            <WorkbenchAction
              kind="ghost"
              size="compact"
              className="h-7 px-2 text-muted-foreground"
              disabled={busy}
              title="Rerun without cache"
              onClick={() =>
                void runVerb("Rerun fresh", () =>
                  workspaceApi.rerunRun(projectId, experimentId, runId, true),
                )
              }
            >
              Fresh
            </WorkbenchAction>
          </>
        )}

        {showCancel && (
          <WorkbenchAction
            kind="secondary"
            size="compact"
            className="h-7 gap-1 text-destructive hover:bg-destructive/10 hover:text-destructive"
            onClick={() => {
              void onCancel();
            }}
          >
            <Ban className="h-3.5 w-3.5" />
            Cancel
          </WorkbenchAction>
        )}

        {/* ── Knowledge (outcome only) ────────────────────────────────── */}
        {showHarvest && (
          <HarvestDialog
            projectId={projectId}
            experimentId={experimentId}
            runId={runId}
            onHarvested={onHarvested}
          />
        )}

        {/* ── Utilities ───────────────────────────────────────────────── */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <WorkbenchIconAction label="More" kind="ghost" className="h-7 w-7" aria-label="More">
              <MoreHorizontal className="h-4 w-4" />
            </WorkbenchIconAction>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-40">
            <DropdownMenuItem asChild>
              <a
                href={exportUrl}
                download={`run-${runId}.zip`}
                onClick={() => toast("Downloading…")}
              >
                <Download className="h-3.5 w-3.5" />
                Export
              </a>
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => {
                void navigator.clipboard.writeText(runId);
                toast.success("Copied");
              }}
            >
              <Copy className="h-3.5 w-3.5" />
              Copy ID
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={onOpenAgent}>
              <Bot className="h-3.5 w-3.5" />
              Agent
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <AlertDialog open={verbError !== null} onOpenChange={(open) => !open && setVerbError(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Failed</AlertDialogTitle>
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

export { POST_DISPATCH_TAB };
