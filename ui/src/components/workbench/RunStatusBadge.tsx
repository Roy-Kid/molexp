/**
 * Product surface for run / execution status.
 * Domain prop only — shadcn Badge API stays inside StatusBadge.
 */

import type { JSX } from "react";

import { StatusBadge, type StatusBadgeSize } from "@/app/components/entity";
import { type CanonicalStatus, canonicalStatusFor } from "@/app/components/entity/status";

export type RunStatus = CanonicalStatus;

/** Normalize wire aliases once, at the product-component boundary. */
export const normalizeRunStatus = (status: string | null | undefined): RunStatus | null => {
  return canonicalStatusFor(status);
};

export interface RunStatusBadgeProps {
  /** Raw API status is accepted only so this boundary can canonicalize it. */
  status: string | null | undefined;
  size?: StatusBadgeSize;
  dot?: boolean;
  showLabel?: boolean;
}

export const RunStatusBadge = ({
  status,
  size = "sm",
  dot = true,
  showLabel = true,
}: RunStatusBadgeProps): JSX.Element | null => {
  const canonicalStatus = normalizeRunStatus(status);
  return <StatusBadge status={canonicalStatus} size={size} dot={dot} showLabel={showLabel} />;
};
