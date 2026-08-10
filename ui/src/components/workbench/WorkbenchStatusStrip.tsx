/**
 * Persistent bottom status bar — MolVis chrome:
 * ``h-statusbar`` + ``border-t border-border/80 bg-background``,
 * heartbeat lamp + activity line + **always-visible progress while loading**.
 */

import { AlertCircle, AlertTriangle, Check, Info, Loader2 } from "lucide-react";
import { type JSX, useEffect, useRef, useState } from "react";

import { useSyncPulse } from "@/app/state/syncPulse";
import { Button } from "@/components/ui/button";
import { useStatusMessage } from "@/hooks/useStatusMessage";
import { reportStatus, type StatusReportType } from "@/lib/status-report";
import { cn } from "@/lib/utils";

import { WorkbenchHeartbeat } from "./WorkbenchHeartbeat";

export interface WorkbenchStatusStripProps {
  className?: string;
  /** Override the pulse generation (tests / storybook). Defaults to global sync pulse. */
  beat?: number;
  /** Manual workspace refresh, triggered by the heartbeat. */
  onRefresh?: () => void;
  isRefreshing?: boolean;
  /**
   * Called once when a remote index walk finishes (phase → done) so the
   * shell can re-fetch projects after a background link/refresh.
   */
  onRemoteIndexReady?: () => void;
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
    return <Loader2 className={cn(className, "mol-motion-progress-spin text-status-running-foreground")} />;
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
  return (
    body.phase === "counting" ||
    body.phase === "fetching" ||
    body.indexing === true
  );
}

export const WorkbenchStatusStrip = ({
  className,
  beat: beatProp,
  onRefresh,
  isRefreshing = false,
  onRemoteIndexReady,
}: WorkbenchStatusStripProps): JSX.Element => {
  const pulse = useSyncPulse();
  const beat = beatProp ?? pulse;
  const { activity, dismissActivity } = useStatusMessage();
  const [cacheStatus, setCacheStatus] = useState<CacheStatus | null>(null);
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
          schedule(2000);
          return;
        }
        const body = (await res.json()) as CacheStatus;
        if (cancelled) return;

        const active = isRemoteIndexing(body);
        setCacheStatus(body.cached ? body : null);

        // Mirror remote progress onto the status bus so the line + bar stay in
        // lockstep (and so other consumers see the same tip).
        if (body.cached && active) {
          // counting → progress 0 (keeps the line alive, indeterminate bar)
          // fetching → determinate percent from done/total
          const pct =
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
          }
        } else if (wasIndexing.current && body.phase === "done") {
          lastReportedPhase.current = "done";
          reportStatus("Remote index ready", "success");
          onRemoteIndexReady?.();
        }

        wasIndexing.current = active;
        schedule(active ? 200 : 2000);
      } catch {
        if (!cancelled) setCacheStatus(null);
        schedule(2500);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }, [onRemoteIndexReady]);

  // Generic workspace bootstrap / manual refresh — indeterminate progress so
  // the bar is never blank while loading (even without a remote cache).
  useEffect(() => {
    if (!isRefreshing) return;
    if (isRemoteIndexing(cacheStatus)) return; // remote path owns the message
    reportStatus("Loading workspace…", "info", 0);
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

  // Determinate only when we know a percent; counting / generic load → indeterminate.
  const lineProgress: number | undefined = hasBusActivity
    ? activity.progress
    : remotePercent;

  const showProgressTrack = loading || lineProgress !== undefined;
  const indeterminate =
    loading &&
    (lineProgress === undefined ||
      cacheStatus?.phase === "counting" ||
      (showSyncing && !remoteBusy));

  const isAlert = hasBusActivity && (activity.type === "error" || activity.type === "warning");
  const busy = Boolean(isRefreshing || remoteBusy);

  return (
    <div
      className={cn(
        // MolVis: h-statusbar + border-border/80 + bg-background
        "flex h-statusbar shrink-0 items-center border-t border-border/80 bg-background",
        className,
      )}
    >
      <div className="flex h-full shrink-0 items-center px-1">
        <WorkbenchHeartbeat
          beat={beat}
          label={
            busy
              ? "Syncing workspace… (click to re-trigger refresh)"
              : "Live sync — click to refresh workspace tree"
          }
          running={busy}
          onClick={onRefresh}
          disabled={!onRefresh}
        />
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
            key={hasBusActivity ? activity.pulse : showRemote ? `remote-${cacheStatus?.phase}` : "syncing"}
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
