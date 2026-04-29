import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import type { JSX, ReactNode } from "react";

import { cn } from "@/lib/utils";

import { Sparkline, type SparklineTrend } from "./charts/Sparkline";

export type KpiTone = "running" | "pending" | "failed" | "succeeded" | "neutral";

const DOT_CLASS: Record<KpiTone, string> = {
  running: "bg-info",
  pending: "bg-muted-foreground/50",
  failed: "bg-destructive",
  succeeded: "bg-success",
  neutral: "bg-muted-foreground/40",
};

const SPARK_TREND: Record<KpiTone, SparklineTrend> = {
  running: "up",
  pending: "flat",
  failed: "down",
  succeeded: "up",
  neutral: "flat",
};

const SPARK_STROKE: Record<KpiTone, string> = {
  running: "stroke-info",
  pending: "stroke-muted-foreground",
  failed: "stroke-destructive",
  succeeded: "stroke-success",
  neutral: "stroke-muted-foreground",
};

const SPARK_FILL: Record<KpiTone, string> = {
  running: "fill-info/15",
  pending: "fill-muted-foreground/10",
  failed: "fill-destructive/15",
  succeeded: "fill-success/15",
  neutral: "fill-muted-foreground/10",
};

export interface RunsKpiCardProps {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  delta?: number | null;
  /** Suffix shown after the delta number (e.g. "in last hour"). */
  deltaSuffix?: string;
  /** Inverts the colour mapping — useful for "Failed" where +1 is bad. */
  invertDelta?: boolean;
  sparkline?: number[];
  tone?: KpiTone;
}

const formatDelta = (delta: number): string => {
  const abs = Math.abs(delta);
  if (delta > 0) return `+${abs}`;
  if (delta < 0) return `−${abs}`;
  return "0";
};

const deltaToneClass = (delta: number, invert: boolean): string => {
  if (delta === 0) return "text-muted-foreground";
  const positive = delta > 0;
  const good = invert ? !positive : positive;
  return good ? "text-success" : "text-destructive";
};

export const RunsKpiCard = ({
  label,
  value,
  detail,
  delta,
  deltaSuffix,
  invertDelta = false,
  sparkline,
  tone = "neutral",
}: RunsKpiCardProps): JSX.Element => {
  const hasSpark = sparkline && sparkline.length > 0 && sparkline.some((v) => v > 0);
  const showDelta = typeof delta === "number" && Number.isFinite(delta);

  return (
    <div className="flex h-full flex-col gap-2 rounded-md border border-border/60 bg-card p-3">
      <div className="flex items-center gap-1.5">
        <span
          aria-hidden="true"
          className={cn("inline-block h-1.5 w-1.5 rounded-full", DOT_CLASS[tone])}
        />
        <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
      </div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="min-w-0 break-words text-2xl font-semibold leading-none tabular-nums text-foreground">
          {value}
        </span>
        {hasSpark && (
          <div className="h-7 w-20 shrink-0 opacity-90">
            <Sparkline
              data={sparkline}
              trend={SPARK_TREND[tone]}
              strokeClassName={SPARK_STROKE[tone]}
              fillClassName={SPARK_FILL[tone]}
              ariaLabel={`${label} trend`}
            />
          </div>
        )}
      </div>
      {(showDelta || detail) && (
        <div className="flex items-center justify-between text-[11px] text-muted-foreground">
          {showDelta ? (
            <span
              className={cn(
                "inline-flex items-center gap-0.5 font-medium tabular-nums",
                deltaToneClass(delta as number, invertDelta),
              )}
            >
              {(delta as number) > 0 ? (
                <ArrowUp className="h-3 w-3" />
              ) : (delta as number) < 0 ? (
                <ArrowDown className="h-3 w-3" />
              ) : (
                <Minus className="h-3 w-3" />
              )}
              {formatDelta(delta as number)}
              {deltaSuffix && <span className="ml-1 text-muted-foreground">{deltaSuffix}</span>}
            </span>
          ) : (
            <span />
          )}
          {detail && <span className="truncate">{detail}</span>}
        </div>
      )}
    </div>
  );
};
