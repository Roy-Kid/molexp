import { WorkbenchAction, WorkbenchIconAction, WorkbenchTag } from "@/components/workbench";
// ─────────────────────────────────────────────────────────────────────────────
// Dashboard primitives — section / chart vocabulary for entity Overviews.
// Shared by Project / Experiment / Run so hierarchy pages share one surface
// language (canvas shell + mono section headers + hairline borders). Pure
// presentation only; no shadcn Card layout wrappers.
// ─────────────────────────────────────────────────────────────────────────────

import { Check, Copy, type LucideIcon } from "lucide-react";
import { type JSX, type ReactNode, useId, useState } from "react";

import { STATUS_GROUPS } from "@/app/runs/statusGroups";
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
  /** Raw value copied by the always-visible copy affordance. */
  copyValue?: string;
}

interface CopyButtonProps {
  value: string;
  label?: string;
  className?: string;
}

/** Compact, reusable copy control for dense scientific data surfaces. */
export const CopyButton = ({ value, label = "value", className }: CopyButtonProps): JSX.Element => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (): Promise<void> => {
    await navigator.clipboard.writeText(value);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  const Icon = copied ? Check : Copy;
  return (
    <WorkbenchIconAction
      label={copied ? `${label} copied` : `Copy ${label}`}
      className={cn("size-6 text-muted-foreground", copied && "text-success", className)}
      onClick={(event) => {
        event.stopPropagation();
        void handleCopy();
      }}
    >
      <Icon className="size-3" aria-hidden />
    </WorkbenchIconAction>
  );
};

/** One labeled field on a dashboard — sentence-case label, quiet hierarchy. */
export const MetaField = ({
  label,
  value,
  mono = false,
  className,
  title,
  copyValue,
}: MetaFieldProps): JSX.Element => (
  <div className={cn("min-w-0", className)}>
    <dt className="font-mono text-micro uppercase tracking-wider text-muted-foreground">{label}</dt>
    <dd
      className={cn(
        "mt-0.5 flex min-w-0 items-center gap-1 text-body text-foreground",
        mono && "font-mono text-label",
      )}
      title={title}
    >
      <span className="min-w-0 truncate">{value}</span>
      {copyValue !== undefined && <CopyButton value={copyValue} label={label} />}
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
      "grid gap-x-4 gap-y-3",
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

/**
 * Headline number for rare decision-bearing metrics (e.g. filterable run
 * dashboards). Prefer MetaStrip / StatusInline on Project · Experiment · Run
 * overviews — see information-design stage `info`.
 */
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
          className={cn("inline-block h-1.5 w-1.5 shrink-0 rounded-full", STAT_DOT_TONE[tone])}
        />
        <span className="truncate font-mono text-micro uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
      </div>
      <div
        className={cn(
          "mt-1.5 text-heading font-semibold leading-none tracking-tight tabular-nums",
          muted ? "text-muted-foreground/45" : STAT_VALUE_TONE[tone],
        )}
      >
        {value}
      </div>
      {hint != null && hint !== "" && (
        <div className="mt-1 truncate text-micro text-muted-foreground">{hint}</div>
      )}
    </>
  );

  const shell = "flex min-h-18 h-full flex-col px-3 py-2.5 text-left transition-colors";

  if (onClick) {
    return (
      <WorkbenchAction
        kind="ghost"
        size="content"
        type="button"
        onClick={onClick}
        className={cn(
          shell,
          "hover:bg-interactive/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
          active && "bg-accent-muted/60",
        )}
      >
        {body}
      </WorkbenchAction>
    );
  }
  return <div className={shell}>{body}</div>;
};

interface StatGridProps {
  children: ReactNode;
  className?: string;
}

/** Responsive grid for a row of StatCard. Prefer MetaStrip on entity overviews. */
export const StatGrid = ({ children, className }: StatGridProps): JSX.Element => (
  <div
    className={cn(
      "grid grid-cols-2 border-b border-border/70 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6",
      className,
    )}
  >
    {children}
  </div>
);

// ── MetaStrip — dense fact row (settings / secondary panels only) ────────────

export interface MetaStripItem {
  label: string;
  value: ReactNode;
  mono?: boolean;
  copyValue?: string;
  tone?: StatTone;
  /** Hide when value is empty/zero noise. */
  hide?: boolean;
}

interface MetaStripProps {
  items: MetaStripItem[];
  className?: string;
  /** Optional trailing control cluster (actions, status bar). */
  trailing?: ReactNode;
}

/**
 * Compact fact row for settings and secondary panels.
 * Entity overviews should prefer OverviewToolbar + a primary table instead.
 */
export const MetaStrip = ({ items, className, trailing }: MetaStripProps): JSX.Element => {
  const visible = items.filter((item) => !item.hide);
  return (
    <div
      className={cn(
        "flex min-h-control-comfortable flex-wrap items-center gap-x-1 gap-y-1 border-b border-border bg-surface-subtle/40 px-3 py-1.5",
        className,
      )}
    >
      <dl className="flex min-w-0 flex-1 flex-wrap items-center gap-x-0 gap-y-1">
        {visible.map((item, index) => (
          <div key={item.label} className="flex min-w-0 items-center">
            {index > 0 && <span className="mx-2 h-3 w-px shrink-0 bg-border" aria-hidden />}
            <dt className="mr-1.5 shrink-0 text-micro text-muted-foreground">{item.label}</dt>
            <dd
              className={cn(
                "flex min-w-0 items-center gap-0.5 text-label text-foreground",
                item.mono && "font-mono",
                item.tone && item.tone !== "neutral" && STAT_VALUE_TONE[item.tone],
                item.tone === "neutral" && "tabular-nums",
              )}
            >
              <span className="min-w-0 truncate">{item.value}</span>
              {item.copyValue !== undefined && (
                <CopyButton value={item.copyValue} label={item.label} className="size-5" />
              )}
            </dd>
          </div>
        ))}
      </dl>
      {trailing != null && (
        <div className="ml-auto flex shrink-0 items-center gap-2">{trailing}</div>
      )}
    </div>
  );
};

// ── OverviewToolbar — single chrome band above the primary table ─────────────

interface OverviewToolbarProps {
  /** Status rollup, fixed-param chips, filters. */
  leading?: ReactNode;
  /** Counts, “Open workflow”, etc. */
  trailing?: ReactNode;
  className?: string;
}

/**
 * One 36px band under entity tabs. Overview pages use this instead of stacking
 * MetaStrip + StatusInline + constants + section headers.
 */
export const OverviewToolbar = ({
  leading,
  trailing,
  className,
}: OverviewToolbarProps): JSX.Element => (
  <div
    className={cn(
      "flex h-9 shrink-0 items-center gap-3 border-b border-border bg-surface px-3",
      className,
    )}
  >
    <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-3 gap-y-1">{leading}</div>
    {trailing != null && <div className="ml-auto flex shrink-0 items-center gap-2">{trailing}</div>}
  </div>
);

// ── StatusInline — one-line status bar + counts (not a legend card) ──────────

interface StatusInlineProps {
  counts: StatusCountRollup;
  className?: string;
  /** Drop own border/padding when nested in OverviewToolbar. */
  embedded?: boolean;
}

/** Compact status: thin bar + non-zero counts only. Empty → null. */
export const StatusInline = ({
  counts,
  className,
  embedded = false,
}: StatusInlineProps): JSX.Element | null => {
  if (counts.total === 0) return null;
  return (
    <div
      className={cn(
        "flex items-center gap-3",
        !embedded && "min-h-control-compact border-b border-border px-3 py-1.5",
        className,
      )}
    >
      <div className="h-1.5 w-24 shrink-0 overflow-hidden rounded-full bg-muted sm:w-32">
        <div className="flex h-full w-full">
          {STATUS_GROUPS.map((group) => {
            const value = counts[group.id];
            if (value === 0) return null;
            return (
              <div
                key={group.id}
                title={`${group.label}: ${value}`}
                className="h-full min-w-0.5"
                style={{
                  width: `${(value / counts.total) * 100}%`,
                  backgroundColor: group.color,
                }}
              />
            );
          })}
        </div>
      </div>
      <ul className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-micro tabular-nums text-muted-foreground">
        {STATUS_GROUPS.map((group) => {
          const value = counts[group.id];
          if (value === 0) return null;
          return (
            <li key={group.id} className="inline-flex items-center gap-1">
              <span
                aria-hidden
                className="size-1.5 shrink-0 rounded-full"
                style={{ backgroundColor: group.color }}
              />
              <span className="font-medium text-foreground">{value}</span>
              <span className="hidden sm:inline">{group.label}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
};

// ── OverviewSurface ──────────────────────────────────────────────────────────

interface OverviewSurfaceProps {
  children: ReactNode;
  className?: string;
  /** Extra classes on the fill surface. */
  surfaceClassName?: string;
}

/**
 * Entity work surface shell. Solid background (no instrumentation grid).
 * Inventory tabs fill height; Overview dashboards scroll with air.
 */
export const OverviewSurface = ({
  children,
  className,
  surfaceClassName,
}: OverviewSurfaceProps): JSX.Element => (
  <div
    className={cn(
      "molexp-dashboard flex min-h-0 flex-1 flex-col overflow-auto bg-background",
      className,
    )}
  >
    <div className={cn("min-h-0 min-w-0 flex-1", surfaceClassName)}>{children}</div>
  </div>
);

/**
 * Padded, max-width canvas — Overview posture and inventory tabs share this air.
 */
export const DashboardCanvas = ({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}): JSX.Element => (
  <div className={cn("mx-auto w-full max-w-5xl space-y-10 px-6 py-8 md:px-10 md:py-10", className)}>
    {children}
  </div>
);

/**
 * Inventory tab body: same padding language as Overview, wider for tables.
 * Full-height tables pass ``fill`` and put the table in a flex child.
 */
export const InventoryCanvas = ({
  children,
  className,
  fill = false,
}: {
  children: ReactNode;
  className?: string;
  /** Stretch to fill the tab (for DataTable tabs). */
  fill?: boolean;
}): JSX.Element => (
  <div
    className={cn(
      fill
        ? "flex h-full min-h-0 w-full flex-col px-4 pt-4 md:px-6 md:pt-5"
        : "mx-auto w-full max-w-6xl space-y-6 px-6 py-6 md:px-10 md:py-8",
      className,
    )}
  >
    {children}
  </div>
);

// ── DashboardCard ────────────────────────────────────────────────────────────

interface DashboardCardProps {
  title?: ReactNode;
  /** Quiet secondary line under the title — prefer short counts, not prose. */
  description?: ReactNode;
  /** Right-aligned header slot — a count, a control, a link. */
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  /** Soft destructive surface for error banners. */
  variant?: "default" | "destructive";
  /** Optional section glyph (Run overview style). */
  icon?: LucideIcon;
  /** Optional count badge in the section header. */
  count?: number;
  /** Copy payload for the header copy control. */
  copyText?: string;
  copyLabel?: string;
}

/**
 * Quiet section on a full-bleed work surface.
 * Toolbar-height label row — no accent gradients, no card chrome.
 */
export const DashboardCard = ({
  title,
  description,
  action,
  children,
  className,
  bodyClassName,
  variant = "default",
  icon: Icon,
  count,
  copyText,
  copyLabel,
}: DashboardCardProps): JSX.Element => {
  const headingId = useId();
  const hasHeader =
    title != null || action != null || count !== undefined || copyText !== undefined;

  return (
    <section
      aria-labelledby={title != null ? headingId : undefined}
      aria-label={typeof title === "string" ? title : undefined}
      className={cn(
        "flex min-w-0 flex-col",
        variant === "destructive" && "bg-status-failed-soft",
        className,
      )}
    >
      {hasHeader && (
        <header className="flex h-control-compact items-center gap-2 border-b border-border px-3">
          {Icon != null && (
            <Icon
              className={cn(
                "size-3.5 shrink-0 text-muted-foreground",
                variant === "destructive" && "text-status-failed",
              )}
              aria-hidden
            />
          )}
          {title != null && (
            <h3
              id={headingId}
              className={cn(
                "min-w-0 truncate text-label font-medium text-foreground",
                variant === "destructive" && "text-status-failed-foreground",
              )}
            >
              {title}
            </h3>
          )}
          {count !== undefined && (
            <span className="font-mono text-micro tabular-nums text-muted-foreground">{count}</span>
          )}
          {description != null && (
            <span className="hidden min-w-0 truncate text-micro text-muted-foreground sm:inline">
              {description}
            </span>
          )}
          <div className="ml-auto flex shrink-0 items-center gap-1">
            {copyText !== undefined && (
              <CopyButton
                value={copyText}
                label={copyLabel ?? String(title ?? "section")}
                className="size-5"
              />
            )}
            {action}
          </div>
        </header>
      )}
      <div className={cn("min-w-0", bodyClassName ?? "px-3 py-2")}>{children}</div>
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
    <div className={cn("space-y-2.5", className)}>
      <div
        className="flex h-1.5 overflow-hidden rounded-control bg-muted"
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
                className="h-full min-w-0.5 transition-[width]"
                style={{
                  width: `${(value / counts.total) * 100}%`,
                  backgroundColor: group.color,
                }}
              />
            );
          })}
      </div>
      {legend && (
        <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3">
          {STATUS_GROUPS.map((group) => {
            const value = counts[group.id];
            return (
              <li key={group.id} className="flex items-center justify-between gap-2 text-label">
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
            <li key={seg.label} className="flex items-center gap-2 text-label">
              <span
                aria-hidden="true"
                className="inline-block h-2 w-2 flex-none rounded-full"
                style={{ backgroundColor: seg.color }}
              />
              <span className="min-w-0 flex-1 truncate text-muted-foreground">{seg.label}</span>
              <span className="font-medium tabular-nums text-foreground">{seg.value}</span>
              <span className="w-control-comfortable text-right tabular-nums text-muted-foreground">
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
    return <p className="text-label text-muted-foreground">{emptyLabel}</p>;
  }
  const ceiling = max ?? Math.max(1, ...data.map((d) => d.value));
  return (
    <ul className="space-y-3">
      {data.map((datum) => {
        const pct = Math.max(datum.value > 0 ? 4 : 0, (datum.value / ceiling) * 100);
        const row = (
          <>
            <div className="mb-1 flex items-baseline justify-between gap-2">
              <span className="min-w-0 truncate text-label text-foreground">{datum.label}</span>
              <span className="flex-none text-label tabular-nums text-muted-foreground">
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
              <WorkbenchAction
                kind="ghost"
                size="content"
                type="button"
                onClick={datum.onClick}
                className="block w-full text-left transition-opacity hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {row}
              </WorkbenchAction>
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
      "flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-border/60 pt-3 text-label text-muted-foreground",
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
            <WorkbenchAction
              kind="ghost"
              size="content"
              type="button"
              className="truncate hover:text-foreground hover:underline"
              onClick={seg.onClick}
            >
              {seg.label}
            </WorkbenchAction>
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
  /** Optional key; omit for bare value chips in a labeled axis row. */
  name?: string;
  value: string;
  className?: string;
}

/** Compact key=value (or value-only) chip used in param previews. */
export const ParamChip = ({ name, value, className }: ParamChipProps): JSX.Element => (
  <WorkbenchTag
    meaning="metadata"
    className={cn(
      "max-w-40 gap-1 rounded-control border-border/70 bg-muted/30 px-2 py-1 font-normal",
      className,
    )}
    title={name ? `${name}=${value}` : value}
  >
    {name ? <span className="truncate text-muted-foreground">{name}</span> : null}
    <span className="truncate font-mono text-foreground">{value}</span>
  </WorkbenchTag>
);

// ── Layout ───────────────────────────────────────────────────────────────────

interface DashboardGridProps {
  children: ReactNode;
  className?: string;
  /** Classes for the outer canvas shell (padding / overflow). */
  shellClassName?: string;
}

/**
 * Full-bleed surface with an optional 12-col grid for multi-column overviews.
 * Prefer MetaStrip + primary table; avoid stacking equal empty widgets.
 */
export const DashboardGrid = ({
  children,
  className,
  shellClassName,
}: DashboardGridProps): JSX.Element => (
  <OverviewSurface
    className={shellClassName}
    surfaceClassName={cn("grid grid-cols-1 content-start lg:grid-cols-12", className)}
  >
    {children}
  </OverviewSurface>
);
