// ─────────────────────────────────────────────────────────────────────────────
// Dashboard primitives — the card/chart vocabulary every entity Overview is
// built from. The Overview tab is a *summary dashboard*: at-a-glance numbers,
// status mix, and small charts. Prose and lower-level detail belong on the
// later tabs, never here. Everything in this file is pure presentation: SVG
// over chart libraries so it stays deterministic and cheap to render.
// ─────────────────────────────────────────────────────────────────────────────

import type { JSX, ReactNode } from "react";
import { cn } from "@/lib/utils";

export type StatTone = "neutral" | "success" | "error" | "running" | "warning";

const STAT_VALUE_TONE: Record<StatTone, string> = {
  neutral: "text-foreground",
  success: "text-success",
  error: "text-destructive",
  running: "text-info",
  warning: "text-warning",
};

const STAT_DOT_TONE: Record<StatTone, string> = {
  neutral: "bg-muted-foreground/40",
  success: "bg-success",
  error: "bg-destructive",
  running: "bg-info",
  warning: "bg-warning",
};

interface StatCardProps {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: StatTone;
  /** Render the value muted when the metric is zero/empty. */
  muted?: boolean;
  onClick?: () => void;
  active?: boolean;
}

/** One headline number in a card. The atom of every dashboard. */
export const StatCard = ({
  label,
  value,
  hint,
  tone = "neutral",
  muted = false,
  onClick,
  active = false,
}: StatCardProps): JSX.Element => {
  const body = (
    <>
      <div className="flex items-center gap-1.5">
        <span
          aria-hidden="true"
          className={cn("inline-block h-1.5 w-1.5 rounded-full", STAT_DOT_TONE[tone])}
        />
        <span className="truncate text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
      </div>
      <div
        className={cn(
          "mt-1.5 text-2xl font-semibold leading-none tabular-nums",
          muted ? "text-muted-foreground/50" : STAT_VALUE_TONE[tone],
        )}
      >
        {value}
      </div>
      {hint && <div className="mt-1 truncate text-[11px] text-muted-foreground">{hint}</div>}
    </>
  );

  const base = "flex flex-col rounded-lg border bg-card px-3 py-2.5 text-left transition-colors";
  if (onClick) {
    return (
      <button
        type="button"
        onClick={onClick}
        className={cn(
          base,
          "hover:border-foreground/20 hover:bg-muted/40 focus:outline-none focus:ring-2 focus:ring-ring",
          active ? "border-foreground/30 ring-1 ring-inset ring-foreground/20" : "border-border/60",
        )}
      >
        {body}
      </button>
    );
  }
  return <div className={cn(base, "border-border/60")}>{body}</div>;
};

interface StatGridProps {
  children: ReactNode;
  className?: string;
}

/** Responsive grid for a row of :class:`StatCard`. */
export const StatGrid = ({ children, className }: StatGridProps): JSX.Element => (
  <div
    className={cn(
      "grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5",
      className,
    )}
  >
    {children}
  </div>
);

interface DashboardCardProps {
  title?: ReactNode;
  /** Right-aligned header slot — a count, a control, a link. */
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}

/** A titled surface that groups related content on a dashboard. */
export const DashboardCard = ({
  title,
  action,
  children,
  className,
  bodyClassName,
}: DashboardCardProps): JSX.Element => (
  <section className={cn("flex flex-col rounded-lg border border-border/60 bg-card", className)}>
    {(title || action) && (
      <header className="flex items-center justify-between gap-2 border-b border-border/60 px-3 py-2">
        {title && (
          <h3 className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            {title}
          </h3>
        )}
        {action && <div className="flex items-center gap-1">{action}</div>}
      </header>
    )}
    <div className={cn("min-h-0 flex-1 p-3", bodyClassName)}>{children}</div>
  </section>
);

export interface DonutSegment {
  label: string;
  value: number;
  color: string;
}

interface StatusDonutProps {
  segments: DonutSegment[];
  size?: number;
  thickness?: number;
  /** Big number drawn in the hole; defaults to the segment total. */
  centerValue?: ReactNode;
  centerLabel?: ReactNode;
}

/**
 * A donut chart of categorical counts with a centered total and a legend.
 * Built from stroke-dashoffset arcs — no chart library, no layout thrash.
 */
export const StatusDonut = ({
  segments,
  size = 132,
  thickness = 14,
  centerValue,
  centerLabel,
}: StatusDonutProps): JSX.Element => {
  const total = segments.reduce((sum, seg) => sum + seg.value, 0);
  const radius = (size - thickness) / 2;
  const circ = 2 * Math.PI * radius;
  const center = size / 2;
  const visible = segments.filter((seg) => seg.value > 0);

  let acc = 0;

  return (
    <div className="flex items-center gap-4">
      <div className="relative flex-none" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          role="img"
          aria-label={`Status distribution across ${total} items`}
        >
          <g transform={`rotate(-90 ${center} ${center})`}>
            <circle
              cx={center}
              cy={center}
              r={radius}
              fill="none"
              className="stroke-muted"
              strokeWidth={thickness}
            />
            {total > 0 &&
              visible.map((seg) => {
                const frac = seg.value / total;
                const dash = frac * circ;
                const node = (
                  <circle
                    key={seg.label}
                    cx={center}
                    cy={center}
                    r={radius}
                    fill="none"
                    stroke={seg.color}
                    strokeWidth={thickness}
                    strokeDasharray={`${dash} ${circ - dash}`}
                    strokeDashoffset={-acc * circ}
                    strokeLinecap="butt"
                  />
                );
                acc += frac;
                return node;
              })}
          </g>
        </svg>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-semibold leading-none tabular-nums text-foreground">
            {centerValue ?? total}
          </span>
          {centerLabel && (
            <span className="mt-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
              {centerLabel}
            </span>
          )}
        </div>
      </div>
      <ul className="min-w-0 flex-1 space-y-1.5">
        {segments.map((seg) => {
          const pct = total > 0 ? (seg.value / total) * 100 : 0;
          return (
            <li key={seg.label} className="flex items-center gap-2 text-xs">
              <span
                aria-hidden="true"
                className="inline-block h-2.5 w-2.5 flex-none rounded-sm"
                style={{ backgroundColor: seg.color }}
              />
              <span className="min-w-0 flex-1 truncate text-muted-foreground">{seg.label}</span>
              <span className="font-semibold tabular-nums text-foreground">{seg.value}</span>
              <span className="w-9 text-right tabular-nums text-muted-foreground">
                {pct.toFixed(0)}%
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
};

export interface MiniBarDatum {
  label: string;
  value: number;
  hint?: ReactNode;
  color?: string;
  onClick?: () => void;
}

interface MiniBarsProps {
  data: MiniBarDatum[];
  /** Override the axis max; defaults to the largest value present. */
  max?: number;
  emptyLabel?: string;
}

/** A compact horizontal bar list — categorical magnitudes without an axis. */
export const MiniBars = ({ data, max, emptyLabel = "No data." }: MiniBarsProps): JSX.Element => {
  if (data.length === 0) {
    return <p className="text-xs italic text-muted-foreground">{emptyLabel}</p>;
  }
  const ceiling = max ?? Math.max(1, ...data.map((d) => d.value));
  return (
    <ul className="space-y-2">
      {data.map((datum) => {
        const pct = Math.max(datum.value > 0 ? 4 : 0, (datum.value / ceiling) * 100);
        const row = (
          <>
            <div className="mb-1 flex items-baseline justify-between gap-2">
              <span className="min-w-0 truncate text-xs text-foreground">{datum.label}</span>
              <span className="flex-none text-[11px] tabular-nums text-muted-foreground">
                {datum.hint ?? datum.value}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${pct}%`,
                  backgroundColor: datum.color ?? "currentColor",
                }}
              />
            </div>
          </>
        );
        return (
          <li key={datum.label} className={datum.onClick ? undefined : "text-foreground/70"}>
            {datum.onClick ? (
              <button
                type="button"
                onClick={datum.onClick}
                className="block w-full text-left text-foreground/70 transition-opacity hover:opacity-80 focus:outline-none"
              >
                {row}
              </button>
            ) : (
              row
            )}
          </li>
        );
      })}
    </ul>
  );
};

interface DashboardGridProps {
  children: ReactNode;
  className?: string;
}

/** The scroll container + responsive 12-col grid the Overview lays cards onto. */
export const DashboardGrid = ({ children, className }: DashboardGridProps): JSX.Element => (
  <div className="flex-1 overflow-auto">
    <div className={cn("grid grid-cols-1 gap-3 p-4 lg:grid-cols-12", className)}>{children}</div>
  </div>
);
