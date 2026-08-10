/**
 * Essential "busy / running" indicator.
 *
 * Pure CSS ring (not a Lucide SVG) so:
 * - macOS Reduce Motion cannot freeze it via layered `* { animation:none }`
 * - React re-renders do not restart a half-drawn SVG stroke animation
 * - transform always applies to a simple block box
 */

import type { JSX } from "react";
import { cn } from "@/lib/utils";

export const ProgressSpinner = ({
  className,
  size = "sm",
  label = "Loading",
}: {
  className?: string;
  size?: "sm" | "md";
  /** Accessible name when the parent does not already label the busy state. */
  label?: string;
}): JSX.Element => (
  <span
    role="status"
    aria-label={label}
    className={cn(
      "mol-progress-spinner",
      size === "md" ? "mol-progress-spinner-md" : "mol-progress-spinner-sm",
      className,
    )}
  />
);
