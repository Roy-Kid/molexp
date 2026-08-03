/**
 * Compact mono status strip for the workbench chrome (28px).
 * Not a toast channel — live scope facts only.
 *
 * Prefer the bottom-panel tab strip (heartbeat + Logs tabs on one row).
 * This strip remains available for secondary footers / tests.
 */

import type { JSX, ReactNode } from "react";

import { useSyncPulse } from "@/app/state/syncPulse";
import { cn } from "@/lib/utils";

import { WorkbenchHeartbeat } from "./WorkbenchHeartbeat";

export interface WorkbenchStatusStripProps {
  children?: ReactNode;
  className?: string;
  /** Override the pulse generation (tests / storybook). Defaults to global sync pulse. */
  beat?: number;
}

export const WorkbenchStatusStrip = ({
  children,
  className,
  beat: beatProp,
}: WorkbenchStatusStripProps): JSX.Element => {
  const pulse = useSyncPulse();
  const beat = beatProp ?? pulse;

  return (
    <div
      role="status"
      aria-label="Workbench status"
      className={cn(
        "flex h-7 flex-none items-center gap-3 border-t border-border bg-background px-3 font-mono text-micro text-muted-foreground tabular-nums",
        className,
      )}
    >
      <WorkbenchHeartbeat beat={beat} />
      {children}
    </div>
  );
};
