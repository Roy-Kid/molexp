// ─────────────────────────────────────────────────────────────────────────────
// Dashboard primitives — section / chart vocabulary for entity Overviews.
// Pure presentation only; no shadcn Card layout wrappers.
// ─────────────────────────────────────────────────────────────────────────────

import { type JSX, type ReactNode, useId } from "react";

import { STATUS_GROUPS } from "@/app/runs/statusGroups";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

/** Minimal status rollup shape (mirrors RunStatusCounts without importing it). */
export interface StatusCountRollup {
  total: number;
  running: number;
  pending: number;
  succeeded: number;
  failed: number;
  cancelled: number;
}

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

// ── MetaField ────────────────────────────────────────────────────────────────

interface MetaFieldProps {
  label: string;
  value: ReactNode;
  /** Monospace value (ids, hashes, raw params). */
  mono?: boolean;
  className?: string;
  title?: string;
}

/** One labeled field on a dashboard — sentence-case label, quiet hierarchy. */
export const MetaField = ({
  label,
  value,
  mono = false,
  className,
  title,
}: MetaFieldProps): JSX.Element => (
  <div className={cn("min-w-0", className)}>
    <dt className="text-xs text-muted-foreground">{label}</dt>
    <dd
      className={cn("mt-1 min-w-0 truncate text-sm text-foreground", mono && "font-mono text-xs")}
      title={title}
    >
      {value}
    </dd>
  </div>
);

interface MetaGridProps {
  children: ReactNode;
  columns?: 2 | 3 | 4 | 5;
  className?: string;
}

export const MetaGrid = ({ children, columns = 2, className }: MetaGridProps): JSX.Element => (
  <dl
    className={cn(
      "grid gap-x-6 gap-y-4",
      columns === 2 && "sm:grid-cols-2",
      columns === 3 && "sm:grid-cols-2 lg:grid-cols-3",
      columns === 4 && "sm:grid-cols-2 lg:grid-cols-4",
      columns === 5 && "sm:grid-cols-2 lg:grid-cols-5",
      className,
    )}
  >
    {children}
  </dl>
);

// ── StatCard ─────────────────────────────────────────────────────────────────

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

/** One headline number. The atom of every dashboard. */
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
      <div className="flex items-center gap-2">
        <span
          aria-hidden="true"
          className={cn("inline-block h-1.5 w-1.5 shrink-0 rounded-full", STAT_DOT_TONE[tone])}
        />
        <span className="truncate text-xs text-muted-foreground">{label}</span>
      </div>
      <div
        className={cn(
          "mt-2 text-display font-semibold leading-none tracking-tight tabular-nums",
          muted ? "text-muted-foreground/45" : STAT_VALUE_TONE[tone],
        )}
      >
        {value}
      </div>
      {hint != null && hint !== "" && (
        <div className="mt-2 truncate text-xs text-muted-foreground">{hint}</div>
      )}
    </>
  );

  const shell = "flex h-full flex-col px-3 py-3 text-left transition-colors";

  if (onClick) {
    return (
      <button
        type="button"
        onClick={onClick}
        className={cn(
          shell,
          "rounded-md hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          active && "bg-primary/5 ring-1 ring-inset ring-primary/25",
        )}
      >
        {body}
      </button>
    );
  }
  return <div className={shell}>{body}</div>;
};

interface StatGridProps {
  children: ReactNode;
  className?: string;
}

/** Responsive grid for a row of StatCard. */
export const StatGrid = ({ children, className }: StatGridProps): JSX.Element => (
  <div
    className={cn(
      "grid grid-cols-2 gap-x-3 gap-y-4 border-y border-border/60 py-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5",
      className,
    )}
  >
    {children}
  </div>
);

// ── DashboardCard ────────────────────────────────────────────────────────────

interface DashboardCardProps {
  title?: ReactNode;
  description?: ReactNode;
  /** Right-aligned header slot — a count, a control, a link. */
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  /** Soft destructive surface for error banners. */
  variant?: "default" | "destructive";
}

/** A titled section that groups related content on a dashboard (no Card chrome). */
export const DashboardCard = ({
  title,
  description,
  action,
  children,
  className,
  bodyClassName,
  variant = "default",
}: DashboardCardProps): JSX.Element => {
  const headingId = useId();

  return (
    <section
      aria-labelledby={title != null ? headingId : undefined}
      className={cn(
        "flex flex-col",
        variant === "destructive" && "border-y border-status-failed/30 bg-status-failed-soft px-3",
        className,
      )}
    >
      {(title != null || action != null || description != null) && (
        <header className="flex flex-row items-start justify-between gap-3 border-b border-border/60 pb-2">
          <div className="min-w-0 space-y-1">
            {title != null && (
              <h3 id={headingId} className="text-body font-medium leading-none text-foreground">
                {title}
              </h3>
            )}
            {description != null && (
              <p className="text-label leading-relaxed text-muted-foreground">{description}</p>
            )}
          </div>
          {action != null && <div className="flex-none self-center">{action}</div>}
        </header>
      )}
      <div className={cn("pt-3", bodyClassName)}>{children}</div>
    </section>
  );
};

// ── Status distribution ──────────────────────────────────────────────────────

interface StatusDistributionProps {
  counts: StatusCountRollup;
  /** Show the legend list under the bar. Default true. */
  legend?: boolean;
  className?: string;
}

/** Segmented status bar + optional legend — shared by project / experiment. */
export const StatusDistribution = ({
  counts,
  legend = true,
  className,
}: StatusDistributionProps): JSX.Element => {
  const empty = counts.total === 0;

  return (
    <div className={cn("space-y-3", className)}>
      <div
        className="flex h-2 overflow-hidden rounded-full bg-muted"
        role="img"
        aria-label={empty ? "No runs" : `Status mix across ${counts.total} runs`}
      >
        {!empty &&
          STATUS_GROUPS.map((group) => {
            const value = counts[group.id];
            if (value === 0) return null;
            return (
              <div
                key={group.id}
                title={`${group.label}: ${value}`}
                className="h-full min-w-[2px] transition-[width]"
                style={{
                  width: `${(value / counts.total) * 100}%`,
                  backgroundColor: group.color,
                }}
              />
            );
          })}
      </div>
      {legend && (
        <ul className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3">
          {STATUS_GROUPS.map((group) => {
            const value = counts[group.id];
            return (
              <li key={group.id} className="flex items-center justify-between gap-2 text-xs">
                <span className="inline-flex min-w-0 items-center gap-2 text-muted-foreground">
                  <span
                    aria-hidden="true"
                    className="h-1.5 w-1.5 shrink-0 rounded-full"
                    style={{ backgroundColor: group.color }}
                  />
                  <span className="truncate">{group.label}</span>
                </span>
                <span
                  className={cn(
                    "font-medium tabular-nums text-foreground",
                    value === 0 && "text-muted-foreground/50",
                  )}
                >
                  {value}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};

// ── Charts ───────────────────────────────────────────────────────────────────

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
 * Built from stroke-dashoffset arcs — no chart library.
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
          <span className="text-display font-semibold leading-none tabular-nums text-foreground">
            {centerValue ?? total}
          </span>
          {centerLabel && (
            <span className="mt-1 text-micro text-muted-foreground">{centerLabel}</span>
          )}
        </div>
      </div>
      <ul className="min-w-0 flex-1 space-y-2">
        {segments.map((seg) => {
          const pct = total > 0 ? (seg.value / total) * 100 : 0;
          return (
            <li key={seg.label} className="flex items-center gap-2 text-xs">
              <span
                aria-hidden="true"
                className="inline-block h-2 w-2 flex-none rounded-full"
                style={{ backgroundColor: seg.color }}
              />
              <span className="min-w-0 flex-1 truncate text-muted-foreground">{seg.label}</span>
              <span className="font-medium tabular-nums text-foreground">{seg.value}</span>
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
    return <p className="text-xs text-muted-foreground">{emptyLabel}</p>;
  }
  const ceiling = max ?? Math.max(1, ...data.map((d) => d.value));
  return (
    <ul className="space-y-3">
      {data.map((datum) => {
        const pct = Math.max(datum.value > 0 ? 4 : 0, (datum.value / ceiling) * 100);
        const row = (
          <>
            <div className="mb-1 flex items-baseline justify-between gap-2">
              <span className="min-w-0 truncate text-xs text-foreground">{datum.label}</span>
              <span className="flex-none text-xs tabular-nums text-muted-foreground">
                {datum.hint ?? datum.value}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-foreground/70"
                style={{
                  width: `${pct}%`,
                  ...(datum.color ? { backgroundColor: datum.color } : undefined),
                }}
              />
            </div>
          </>
        );
        return (
          <li key={datum.label}>
            {datum.onClick ? (
              <button
                type="button"
                onClick={datum.onClick}
                className="block w-full text-left transition-opacity hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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

// ── Breadcrumb trail (inline entity path) ────────────────────────────────────

interface EntityPathSegment {
  label: string;
  onClick?: () => void;
}

interface EntityPathProps {
  segments: EntityPathSegment[];
  trailing?: ReactNode;
  className?: string;
}

/** Quiet project / experiment / workflow path under a summary card. */
export const EntityPath = ({ segments, trailing, className }: EntityPathProps): JSX.Element => (
  <div
    className={cn(
      "flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-border/60 pt-3 text-xs text-muted-foreground",
      className,
    )}
  >
    {segments.map((seg, i) => {
      // Stable path key from the prefix labels (unique within a breadcrumb).
      const pathKey = segments
        .slice(0, i + 1)
        .map((s) => s.label)
        .join("/");
      return (
        <span key={pathKey} className="inline-flex items-center gap-2">
          {i > 0 && <span className="text-border">/</span>}
          {seg.onClick ? (
            <button
              type="button"
              className="truncate hover:text-foreground hover:underline"
              onClick={seg.onClick}
            >
              {seg.label}
            </button>
          ) : (
            <span className="truncate">{seg.label}</span>
          )}
        </span>
      );
    })}
    {trailing != null && <span className="ml-auto font-mono text-micro">{trailing}</span>}
  </div>
);

// ── Chip / tag ───────────────────────────────────────────────────────────────

interface ParamChipProps {
  name: string;
  value: string;
  className?: string;
}

/** Compact key=value chip used in run tables and parameter previews. */
export const ParamChip = ({ name, value, className }: ParamChipProps): JSX.Element => (
  <Badge
    variant="outline"
    className={cn(
      "max-w-[140px] gap-1 rounded-md border-border/70 bg-muted/30 px-2 py-1 font-normal",
      className,
    )}
    title={`${name}=${value}`}
  >
    <span className="truncate text-muted-foreground">{name}</span>
    <span className="truncate font-mono text-foreground">{value}</span>
  </Badge>
);

// ── Layout ───────────────────────────────────────────────────────────────────

interface DashboardGridProps {
  children: ReactNode;
  className?: string;
}

/** Scroll container + responsive 12-col grid the Overview lays cards onto. */
export const DashboardGrid = ({ children, className }: DashboardGridProps): JSX.Element => (
  <div className="flex-1 overflow-auto">
    <div className={cn("grid grid-cols-1 gap-4 p-4 md:p-4 lg:grid-cols-12", className)}>
      {children}
    </div>
  </div>
);
