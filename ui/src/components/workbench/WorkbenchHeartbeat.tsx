/**
 * Live sync indicator for the workbench chrome (bottom tab strip).
 * Idle is neutral, active refresh uses running blue, and each completed poll
 * gets one short neutral acknowledgement pulse.
 */

import { HeartPulse } from "lucide-react";
import { type JSX, useEffect, useState } from "react";

import { cn } from "@/lib/utils";

export interface WorkbenchHeartbeatProps {
  /** Monotonic counter — each increment restarts the acknowledgement pulse. */
  beat: number;
  className?: string;
  label?: string;
  running?: boolean;
  /** Optional click — e.g. manual workspace refresh. */
  onClick?: () => void;
  disabled?: boolean;
}

const PULSE_MS = 180;

export const WorkbenchHeartbeat = ({
  beat,
  className,
  label = "Live sync",
  running = false,
  onClick,
  disabled = false,
}: WorkbenchHeartbeatProps): JSX.Element => {
  // Class-toggle (not only remount) so the CSS animation reliably restarts.
  const [pulsing, setPulsing] = useState(false);

  useEffect(() => {
    // Skip the initial mount beat so the icon doesn't flash before first poll.
    if (beat === 0) return;
    // Drop the class for one frame, then re-add so CSS keyframes always restart.
    setPulsing(false);
    const id = window.requestAnimationFrame(() => {
      setPulsing(true);
    });
    const clear = window.setTimeout(() => setPulsing(false), PULSE_MS);
    return () => {
      window.cancelAnimationFrame(id);
      window.clearTimeout(clear);
    };
  }, [beat]);

  const lamp = (
    <span
      className={cn(
        "inline-flex h-4 w-4 flex-none items-center justify-center text-muted-foreground",
        running && "text-status-running-foreground",
        pulsing && "mol-heartbeat-pulse",
        className,
      )}
    >
      <HeartPulse className="h-3.5 w-3.5" aria-hidden strokeWidth={1.5} />
    </span>
  );

  if (!onClick) {
    return (
      <span title={label} aria-label={label} role="status">
        {lamp}
      </span>
    );
  }

  return (
    <button
      type="button"
      className={cn(
        "inline-flex h-control-compact w-control-compact flex-none items-center justify-center rounded-[var(--radius-control)]",
        "text-muted-foreground transition-colors hover:bg-interactive hover:text-foreground",
        "disabled:pointer-events-none disabled:opacity-50",
      )}
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
    >
      {lamp}
    </button>
  );
};
