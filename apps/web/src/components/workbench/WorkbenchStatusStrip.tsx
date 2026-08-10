/**
 * Persistent bottom status bar — MolVis chrome:
 * ``h-statusbar`` + ``border-t border-border/80 bg-background``,
 * heartbeat lamp + activity line + **always-visible progress while loading**.
 *
 * Heartbeat click opens a connection-status popover (workspace identity +
 * link state). Manual refresh lives on the ContextBar / left-panel header —
 * not on the lamp.
 */

import {
  AlertCircle,
  AlertTriangle,
  Check,
  CloudOff,
  HardDrive,
  Info,
  Loader2,
  Server,
} from "lucide-react";
import { type JSX, useEffect, useRef, useState } from "react";

import { pulseSync, useSyncPulse } from "@/app/state/syncPulse";
import type { ServedWorkspaceSummary } from "@/app/types";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { useStatusMessage } from "@/hooks/useStatusMessage";
import { reportStatus, type StatusReportType } from "@/lib/status-report";
import { cn } from "@/lib/utils";

import { WorkbenchHeartbeat } from "./WorkbenchHeartbeat";

export interface WorkbenchStatusStripProps {
  className?: string;
  /** Override the pulse generation (tests / storybook). Defaults to global sync pulse. */
  beat?: number;
  isRefreshing?: boolean;
  /**
   * Called once when a remote index walk finishes (phase → done) so the
   * shell can re-fetch projects after a background link/refresh.
   */
  onRemoteIndexReady?: () => void;
  /** Active served workspace for the connection-status popover. */
  activeWorkspace?: ServedWorkspaceSummary | null;
}

interface CacheStatus {
  cached: boolean;
  ready?: boolean | null;
  indexing?: boolean | null;
  phase: string;
  total: number;
  done: number;
  percent: number | null;
  message: string;
}

function ActivityIcon({
  type,
  progress,
  running,
}: {
  type: StatusReportType;
  progress?: number;
  running?: boolean;
}): JSX.Element {
  const className = "size-3 shrink-0";
  if (type === "error") {
    return <AlertCircle className={cn(className, "text-status-failed")} />;
  }
  if (type === "warning") {
    return <AlertTriangle className={cn(className, "text-status-warning")} />;
  }
  if (type === "success") {
    return <Check className={cn(className, "text-status-completed")} />;
  }
  if (progress !== undefined || running) {
    return (
      <Loader2
        className={cn(className, "mol-motion-progress-spin text-status-running-foreground")}
      />
    );
  }
  return <Info className={cn(className, "text-muted-foreground")} />;
}

function activityTextClass(type: StatusReportType): string {
  switch (type) {
    case "error":
      return "text-status-failed-foreground";
    case "warning":
      return "text-status-warning-foreground";
    case "success":
      return "text-status-completed-foreground";
    default:
      return "text-muted-foreground";
  }
}

/**
 * Loading progress track.
 * - *percent* set → determinate fill
 * - otherwise → indeterminate shimmer (counting / generic sync)
 */
function ProgressTrack({
  percent,
  indeterminate,
}: {
  percent?: number;
  indeterminate?: boolean;
}): JSX.Element {
  const clamped =
    percent !== undefined && Number.isFinite(percent)
      ? Math.max(0, Math.min(100, percent))
      : undefined;

  return (
    <div
      className="relative h-1 min-w-[5.5rem] max-w-[12rem] flex-1 overflow-hidden rounded-full bg-muted sm:min-w-[8rem]"
      role="progressbar"
      aria-valuenow={clamped !== undefined ? Math.round(clamped) : undefined}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-busy={indeterminate || clamped === undefined}
    >
      {clamped !== undefined && !indeterminate ? (
        <div
          className="h-full rounded-full bg-status-running-foreground transition-[width] duration-150 ease-out"
          style={{ width: `${clamped}%` }}
        />
      ) : (
        <div className="mol-status-progress-indeterminate h-full w-1/3 rounded-full bg-status-running-foreground/90" />
      )}
    </div>
  );
}

function isRemoteIndexing(body: CacheStatus | null): boolean {
  if (!body?.cached) return false;
  return body.phase === "counting" || body.phase === "fetching" || body.indexing === true;
}

function connectionHeadline(args: { busy: boolean; unreachable: boolean; hasWorkspace: boolean }): {
  label: string;
  tone: string;
} {
  if (args.unreachable) {
    return { label: "Unreachable", tone: "text-status-failed-foreground" };
  }
  if (args.busy) {
    return { label: "Syncing", tone: "text-status-running-foreground" };
  }
  if (!args.hasWorkspace) {
    return { label: "No workspace", tone: "text-muted-foreground" };
  }
  return { label: "Connected", tone: "text-status-completed-foreground" };
}

export const WorkbenchStatusStrip = ({
  className,
  beat: beatProp,
  isRefreshing = false,
  onRemoteIndexReady,
  activeWorkspace = null,
}: WorkbenchStatusStripProps): JSX.Element => {
  const pulse = useSyncPulse();
  const beat = beatProp ?? pulse;
  const { activity, dismissActivity } = useStatusMessage();
  const [cacheStatus, setCacheStatus] = useState<CacheStatus | null>(null);
  const [statusOpen, setStatusOpen] = useState(false);
  const wasIndexing = useRef(false);
  const lastReportedPhase = useRef<string>("");

  // Poll remote index progress. Never stop the loop on a single failure.
  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;

    const schedule = (ms: number) => {
      timer = window.setTimeout(() => {
        void tick();
      }, ms);
    };

    const tick = async (): Promise<void> => {
      try {
        const res = await fetch("/api/workspace/cache/status");
        if (cancelled) return;
        if (!res.ok) {
          setCacheStatus(null);
          // Poll completed (even failed) — status lamp breathes once.
          pulseSync();
          schedule(2000);
          return;
        }
        const body = (await res.json()) as CacheStatus;
        if (cancelled) return;

        const active = isRemoteIndexing(body);
        setCacheStatus(body.cached ? body : null);

        // Whether this tick should flash the heartbeat lamp.
        // Idle polls (2s) always breathe; the 200ms active loop would thrash
        // the 280ms ack animation, so while indexing we only pulse when
        // phase/progress moves (continuous running CSS covers the rest).
        let shouldPulse = !active;

        // Mirror remote progress onto the status bus so the line + bar stay in
        // lockstep (and so other consumers see the same tip).
        if (body.cached && active) {
          // counting → progress 0 keeps the tip sticky (no auto-clear); the
          // strip treats 0 as indeterminate so the bar never freezes at 0%.
          // fetching → determinate percent from done/total (still coerce 0 →
          // indeterminate until the first file lands).
          const pct: number =
            body.phase === "counting"
              ? 0
              : body.percent != null && Number.isFinite(body.percent)
                ? body.percent
                : body.total > 0
                  ? (100 * body.done) / body.total
                  : 0;
          const msg =
            body.message ||
            (body.phase === "counting"
              ? "Counting remote files…"
              : body.total > 0
                ? `Syncing remote tree (${body.done}/${body.total})…`
                : "Syncing remote tree…");
          // Re-report when phase/progress moves so the bar stays live.
          const sig = `${body.phase}:${body.done}:${body.total}:${Math.round(pct)}`;
          if (sig !== lastReportedPhase.current) {
            lastReportedPhase.current = sig;
            reportStatus(msg, "info", pct);
            shouldPulse = true;
          }
        } else if (wasIndexing.current && body.phase === "done") {
          lastReportedPhase.current = "done";
          reportStatus("Remote index ready", "success");
          onRemoteIndexReady?.();
          shouldPulse = true;
        }

        wasIndexing.current = active;
        // Drive the status-strip heartbeat (same contract as runs / workspace
        // snapshot polls — see syncPulse.ts).
        if (shouldPulse) pulseSync();
        schedule(active ? 200 : 2000);
      } catch {
        if (!cancelled) {
          setCacheStatus(null);
          pulseSync();
        }
        schedule(2500);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }, [onRemoteIndexReady]);

  // Generic workspace bootstrap / manual refresh — keep the strip busy for the
  // full fetch. progress=0 is sticky in the status bus; the strip maps 0 →
  // indeterminate so the bar shimmers instead of freezing as a zero-width fill.
  const wasRefreshing = useRef(false);
  useEffect(() => {
    if (isRemoteIndexing(cacheStatus)) {
      // Remote index owns the line — keep refresh edge tracking in sync so we
      // don't fire a stale "ready" tip when the local fetch ends mid-index.
      wasRefreshing.current = isRefreshing;
      return;
    }
    if (isRefreshing) {
      if (!wasRefreshing.current) {
        // progress=0 keeps the tip sticky (no 5s auto-clear); strip treats 0
        // as indeterminate so the bar shimmers instead of freezing at 0%.
        reportStatus("Loading workspace…", "info", 0);
      }
      wasRefreshing.current = true;
      return;
    }
    if (wasRefreshing.current) {
      wasRefreshing.current = false;
      // Replace the bootstrap tip (auto-clears). Avoid dismissActivity — it
      // would also wipe an unrelated warning/error that landed mid-refresh.
      reportStatus("Workspace ready", "success");
    }
  }, [isRefreshing, cacheStatus]);

  // While a long refresh is in flight, keep the heartbeat lamp breathing even
  // when cache/status is idle (local workspace — no remote index ticks).
  useEffect(() => {
    if (!isRefreshing || isRemoteIndexing(cacheStatus)) return;
    const id = window.setInterval(() => {
      pulseSync();
    }, 1200);
    return () => window.clearInterval(id);
  }, [isRefreshing, cacheStatus]);

  const remoteBusy = isRemoteIndexing(cacheStatus);
  const remotePercent: number | undefined = (() => {
    if (!remoteBusy || !cacheStatus) return undefined;
    if (cacheStatus.phase === "counting") return undefined; // indeterminate
    if (cacheStatus.percent != null && Number.isFinite(cacheStatus.percent)) {
      return cacheStatus.percent;
    }
    if (cacheStatus.total > 0) {
      return (100 * cacheStatus.done) / cacheStatus.total;
    }
    return undefined;
  })();

  const hasBusActivity = activity.text.length > 0;
  const showRemote = remoteBusy;
  const showSyncing = isRefreshing && !remoteBusy;
  const loading = showRemote || showSyncing;

  // Prefer live bus text (includes reportStatus progress suffix); fall back
  // to remote message / generic syncing.
  const lineText = hasBusActivity
    ? activity.text
    : showRemote
      ? cacheStatus?.message ||
        (cacheStatus?.phase === "counting"
          ? "Counting remote files…"
          : cacheStatus && cacheStatus.total > 0
            ? `Syncing remote tree (${cacheStatus.done}/${cacheStatus.total})…`
            : "Syncing remote tree…")
      : showSyncing
        ? "Loading workspace…"
        : "";

  const lineType: StatusReportType = hasBusActivity
    ? activity.type
    : cacheStatus?.phase === "error"
      ? "error"
      : "info";

  // Determinate only when we know a real percent; counting / generic load →
  // indeterminate. Treat progress 0 as "started, unknown fraction" so the bar
  // never freezes as a zero-width determinate fill.
  const rawProgress: number | undefined = hasBusActivity ? activity.progress : remotePercent;
  const lineProgress: number | undefined =
    rawProgress !== undefined && rawProgress > 0 ? rawProgress : undefined;

  const showProgressTrack = loading || lineProgress !== undefined;
  const indeterminate =
    loading &&
    (lineProgress === undefined ||
      cacheStatus?.phase === "counting" ||
      (showSyncing && !remoteBusy));

  const isAlert = hasBusActivity && (activity.type === "error" || activity.type === "warning");
  const busy = Boolean(isRefreshing || remoteBusy);
  const unreachable = Boolean(activeWorkspace?.unreachable);
  const connection = connectionHeadline({
    busy,
    unreachable,
    hasWorkspace: Boolean(activeWorkspace),
  });
  const WorkspaceIcon = unreachable ? CloudOff : activeWorkspace?.isRemote ? Server : HardDrive;

  return (
    <div
      className={cn(
        // MolVis: h-statusbar + border-border/80 + bg-background
        "flex h-statusbar shrink-0 items-center border-t border-border/80 bg-background",
        className,
      )}
    >
      <div className="flex h-full shrink-0 items-center px-1">
        <Popover open={statusOpen} onOpenChange={setStatusOpen}>
          <PopoverTrigger asChild>
            <WorkbenchHeartbeat
              beat={beat}
              label={busy ? "Syncing — click for connection status" : "Connection status"}
              running={busy}
            />
          </PopoverTrigger>
          <PopoverContent side="top" align="start" sideOffset={8} className="w-80 space-y-3 p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="text-label font-medium text-foreground">Connection</p>
                <p className={cn("font-mono text-micro", connection.tone)}>{connection.label}</p>
              </div>
              <span
                className={cn(
                  "inline-flex h-2 w-2 flex-none rounded-full",
                  unreachable
                    ? "bg-status-failed"
                    : busy
                      ? "bg-status-running-foreground mol-motion-progress-pulse"
                      : activeWorkspace
                        ? "bg-status-completed"
                        : "bg-muted-foreground/40",
                )}
                aria-hidden
              />
            </div>

            {activeWorkspace ? (
              <div className="space-y-1.5 rounded-control border border-border/70 bg-muted/30 p-2">
                <div className="flex min-w-0 items-center gap-1.5">
                  <WorkspaceIcon
                    className={cn(
                      "h-3.5 w-3.5 flex-none",
                      unreachable
                        ? "text-status-failed-foreground"
                        : activeWorkspace.isRemote
                          ? "text-status-warning-foreground"
                          : "text-muted-foreground",
                    )}
                    aria-hidden
                  />
                  <span className="min-w-0 truncate font-mono text-micro text-foreground">
                    {activeWorkspace.label}
                  </span>
                </div>
                <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 font-mono text-micro text-muted-foreground">
                  <dt>Kind</dt>
                  <dd className="truncate text-foreground/80">
                    {activeWorkspace.unreachable
                      ? "unreachable"
                      : activeWorkspace.isRemote
                        ? "remote"
                        : "local"}
                  </dd>
                  {activeWorkspace.path && (
                    <>
                      <dt>Path</dt>
                      <dd className="truncate text-foreground/80" title={activeWorkspace.path}>
                        {activeWorkspace.path}
                      </dd>
                    </>
                  )}
                  <dt>Key</dt>
                  <dd className="truncate text-foreground/80" title={activeWorkspace.key}>
                    {activeWorkspace.key}
                  </dd>
                </dl>
              </div>
            ) : (
              <p className="text-micro text-muted-foreground">
                No served workspace is active. Open one from Settings → Remote workspaces, or point
                the server at a local path.
              </p>
            )}

            {(remoteBusy || cacheStatus?.cached) && (
              <div className="space-y-1 font-mono text-micro text-muted-foreground">
                <p className="text-label font-medium text-foreground">Remote index</p>
                <p>
                  {cacheStatus?.phase ?? "idle"}
                  {cacheStatus && cacheStatus.total > 0
                    ? ` · ${cacheStatus.done}/${cacheStatus.total}`
                    : ""}
                  {remotePercent !== undefined ? ` · ${Math.round(remotePercent)}%` : ""}
                </p>
                {cacheStatus?.message ? (
                  <p className="truncate text-foreground/70" title={cacheStatus.message}>
                    {cacheStatus.message}
                  </p>
                ) : null}
              </div>
            )}

            {lineText ? (
              <p className="border-t border-border/60 pt-2 font-mono text-micro text-muted-foreground">
                Activity: <span className={activityTextClass(lineType)}>{lineText}</span>
              </p>
            ) : null}
          </PopoverContent>
        </Popover>
      </div>
      <span className="mx-0.5 h-3.5 w-px shrink-0 bg-border/80" aria-hidden />
      <div
        role="status"
        aria-live={lineType === "error" ? "assertive" : "polite"}
        className="flex h-full min-w-0 flex-1 items-center gap-2 overflow-hidden px-2 font-mono text-micro tabular-nums"
      >
        {lineText && isAlert ? (
          <Button
            key={activity.pulse}
            type="button"
            variant="ghost"
            size="content"
            className="flex min-w-0 max-w-[55%] cursor-pointer justify-start gap-1.5 overflow-hidden rounded-none p-0 text-left font-mono text-micro tabular-nums hover:bg-transparent sm:max-w-[60%]"
            title={`${lineText} (click to dismiss)`}
            onClick={dismissActivity}
          >
            <ActivityIcon type={lineType} progress={lineProgress} />
            <span className={cn("min-w-0 truncate leading-none", activityTextClass(lineType))}>
              {lineText}
            </span>
          </Button>
        ) : lineText ? (
          <div
            key={
              hasBusActivity
                ? activity.pulse
                : showRemote
                  ? `remote-${cacheStatus?.phase}`
                  : "syncing"
            }
            className="flex min-w-0 max-w-[55%] items-center gap-1.5 overflow-hidden sm:max-w-[60%]"
            title={lineText}
          >
            <ActivityIcon type={lineType} progress={lineProgress} running={loading} />
            <span className={cn("min-w-0 truncate leading-none", activityTextClass(lineType))}>
              {lineText}
            </span>
          </div>
        ) : (
          <div className="min-w-0 max-w-[40%]" />
        )}

        {showProgressTrack ? (
          <ProgressTrack percent={lineProgress} indeterminate={indeterminate} />
        ) : null}

        {/* Explicit fraction when we have totals (easier to read than % alone). */}
        {remoteBusy && cacheStatus && cacheStatus.total > 0 && cacheStatus.phase !== "counting" ? (
          <span className="shrink-0 font-mono text-micro tabular-nums text-muted-foreground">
            {cacheStatus.done}/{cacheStatus.total}
          </span>
        ) : lineProgress !== undefined && !indeterminate ? (
          <span className="shrink-0 font-mono text-micro tabular-nums text-muted-foreground">
            {Math.round(lineProgress)}%
          </span>
        ) : null}
      </div>
    </div>
  );
};
