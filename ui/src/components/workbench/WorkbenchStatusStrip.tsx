/** Persistent MolVis-style status bar: heartbeat + one activity line. */

import { AlertCircle, AlertTriangle, Check, Info, Loader2 } from "lucide-react";
import type { JSX } from "react";

import { useSyncPulse } from "@/app/state/syncPulse";
import { Button } from "@/components/ui/button";
import { useStatusMessage } from "@/hooks/useStatusMessage";
import type { StatusReportType } from "@/lib/status-report";
import { cn } from "@/lib/utils";

import { WorkbenchHeartbeat } from "./WorkbenchHeartbeat";

export interface WorkbenchStatusStripProps {
  className?: string;
  /** Override the pulse generation (tests / storybook). Defaults to global sync pulse. */
  beat?: number;
  /** Manual workspace refresh, triggered by the heartbeat. */
  onRefresh?: () => void;
  isRefreshing?: boolean;
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
    return <Loader2 className={cn(className, "mol-motion-progress-spin text-muted-foreground")} />;
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

export const WorkbenchStatusStrip = ({
  className,
  beat: beatProp,
  onRefresh,
  isRefreshing = false,
}: WorkbenchStatusStripProps): JSX.Element => {
  const pulse = useSyncPulse();
  const beat = beatProp ?? pulse;
  const { activity, dismissActivity } = useStatusMessage();
  const hasBusActivity = activity.text.length > 0;
  const showSyncing = !hasBusActivity && isRefreshing;
  const lineText = hasBusActivity ? activity.text : showSyncing ? "Syncing…" : "";
  const lineType: StatusReportType = hasBusActivity ? activity.type : "info";
  const isAlert = hasBusActivity && (activity.type === "error" || activity.type === "warning");

  return (
    <div
      className={cn(
        "flex h-statusbar min-w-0 flex-none items-center gap-1 border-t border-border bg-background px-1",
        className,
      )}
    >
      <WorkbenchHeartbeat
        beat={beat}
        label={isRefreshing ? "Syncing…" : "Live sync — click to refresh"}
        running={isRefreshing}
        onClick={onRefresh}
        disabled={isRefreshing || !onRefresh}
      />
      <span className="mx-1 h-3.5 w-px flex-none bg-border" aria-hidden />
      <div
        role="status"
        aria-live={lineType === "error" ? "assertive" : "polite"}
        className="min-w-0 flex-1 overflow-hidden px-2 font-mono text-micro tabular-nums"
      >
        {lineText && isAlert ? (
          <Button
            key={activity.pulse}
            type="button"
            variant="ghost"
            size="content"
            className="flex w-full min-w-0 cursor-pointer justify-start gap-1.5 overflow-hidden rounded-none p-0 text-left font-mono text-micro tabular-nums hover:bg-transparent"
            title={`${lineText} (click to dismiss)`}
            onClick={dismissActivity}
          >
            <ActivityIcon type={lineType} progress={activity.progress} />
            <span className={cn("min-w-0 truncate leading-none", activityTextClass(lineType))}>
              {lineText}
            </span>
          </Button>
        ) : lineText ? (
          <div
            key={hasBusActivity ? activity.pulse : "syncing"}
            className="flex min-w-0 items-center gap-1.5 overflow-hidden"
            title={lineText}
          >
            <ActivityIcon type={lineType} progress={activity.progress} running={showSyncing} />
            <span className={cn("min-w-0 truncate leading-none", activityTextClass(lineType))}>
              {lineText}
            </span>
          </div>
        ) : null}
      </div>
    </div>
  );
};
