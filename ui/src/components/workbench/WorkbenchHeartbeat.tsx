/**
 * Live sync indicator for the workbench status bar.
 *
 * * Idle — muted but still visible (foreground/60), soft ambient pulse so the
 *   lamp is never "missing" at the bottom of the shell.
 * * Running — running-blue + spin pulse while a refresh / remote index is live.
 * * Click — triggers manual workspace refresh (same as toolbar refresh).
 * * Each completed poll increments *beat* → one short acknowledgement flash.
 */

import { HeartPulse } from "lucide-react";
import { type JSX, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
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

const PULSE_MS = 280;

export const WorkbenchHeartbeat = ({
  beat,
  className,
  label = "Live sync — click to refresh workspace",
  running = false,
  onClick,
  disabled = false,
}: WorkbenchHeartbeatProps): JSX.Element => {
  // Class-toggle (not only remount) so the CSS animation reliably restarts.
  const [pulsing, setPulsing] = useState(false);
  // Local beat so a click still flashes even when the parent beat is unchanged.
  const [localBeat, setLocalBeat] = useState(0);

  useEffect(() => {
    // Skip the initial mount beat so the icon doesn't flash before first poll.
    if (beat === 0 && localBeat === 0) return;
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
  }, [beat, localBeat]);

  const lamp = (
    <span
      className={cn(
        // Idle: readable (not near-invisible muted). Running: status blue.
        "inline-flex h-4 w-4 flex-none items-center justify-center text-foreground/55",
        running && "text-status-running-foreground",
        // Ambient idle pulse so the lamp is always "alive"; stronger on ack.
        !running && "mol-heartbeat-idle",
        pulsing && "mol-heartbeat-pulse",
        running && "mol-motion-progress-pulse",
        className,
      )}
    >
      <HeartPulse className="h-3.5 w-3.5" aria-hidden strokeWidth={1.75} />
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
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      className={cn(
        "text-foreground/55 transition-colors hover:bg-interactive hover:text-foreground",
        running && "text-status-running-foreground",
        // Keep clickable during soft sync so the user can re-trigger refresh;
        // only hard-disable when parent explicitly blocks (e.g. no handler).
        "disabled:pointer-events-none disabled:opacity-40",
      )}
      onClick={() => {
        setLocalBeat((n) => n + 1);
        onClick();
      }}
      disabled={disabled}
      aria-label={label}
      title={label}
    >
      {lamp}
    </Button>
  );
};
