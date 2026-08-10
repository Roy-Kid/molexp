/**
 * Live sync indicator for the workbench status bar.
 *
 * * Idle — muted but still visible (foreground/60), soft ambient pulse so the
 *   lamp is never "missing" at the bottom of the shell.
 * * Running — running-blue + spin pulse while a refresh / remote index is live.
 * * Click — opens the connection-status popover (owned by WorkbenchStatusStrip);
 *   it does **not** trigger a workspace refresh (toolbar / sidebar own refresh).
 * * Each completed poll increments *beat* → one short acknowledgement flash.
 *
 * Forward-ref + button so Radix ``PopoverTrigger asChild`` can own the open
 * gesture without an extra wrapper.
 */

import { HeartPulse } from "lucide-react";
import { type ButtonHTMLAttributes, forwardRef, type JSX, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface WorkbenchHeartbeatProps
  extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children" | "type"> {
  /** Monotonic counter — each increment restarts the acknowledgement pulse. */
  beat: number;
  className?: string;
  label?: string;
  running?: boolean;
}

const PULSE_MS = 280;

export const WorkbenchHeartbeat = forwardRef<HTMLButtonElement, WorkbenchHeartbeatProps>(
  function WorkbenchHeartbeat(
    {
      beat,
      className,
      label = "Connection status",
      running = false,
      disabled = false,
      onClick,
      ...rest
    },
    ref,
  ): JSX.Element {
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

    return (
      <Button
        ref={ref}
        type="button"
        variant="ghost"
        size="icon-sm"
        className={cn(
          "text-foreground/55 transition-colors hover:bg-interactive hover:text-foreground",
          running && "text-status-running-foreground",
          "disabled:pointer-events-none disabled:opacity-40",
          className,
        )}
        onClick={(event) => {
          setLocalBeat((n) => n + 1);
          onClick?.(event);
        }}
        disabled={disabled}
        aria-label={label}
        title={label}
        {...rest}
      >
        <span
          className={cn(
            // Idle: readable (not near-invisible muted). Running: status blue.
            "inline-flex h-4 w-4 flex-none items-center justify-center text-foreground/55",
            running && "text-status-running-foreground",
            // Ambient idle pulse so the lamp is always "alive"; stronger on ack.
            !running && "mol-heartbeat-idle",
            pulsing && "mol-heartbeat-pulse",
            running && "mol-motion-progress-pulse",
          )}
        >
          <HeartPulse className="h-3.5 w-3.5" aria-hidden strokeWidth={1.75} />
        </span>
      </Button>
    );
  },
);
