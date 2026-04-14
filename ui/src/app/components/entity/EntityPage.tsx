import { ChevronLeft, ChevronRight } from "lucide-react";
import type { ComponentType, JSX, ReactNode } from "react";
import { Link } from "react-router-dom";
import type { BreadcrumbItem, SemanticStatus } from "@/app/types";
import { StatusBadge } from "./StatusBadge";

interface SemanticStatusBadgeProps {
  status: SemanticStatus;
}

export const SemanticStatusBadge = ({ status }: SemanticStatusBadgeProps): JSX.Element => {
  return <StatusBadge status={status} />;
};

interface EntityMetricProps {
  label: string;
  value: number | string;
}

export const EntityMetric = ({ label, value }: EntityMetricProps): JSX.Element => {
  return (
    <span className="flex items-baseline gap-1 text-xs">
      <span className="font-semibold tabular-nums text-foreground">{value}</span>
      <span className="text-muted-foreground">{label}</span>
    </span>
  );
};

interface HeaderBreadcrumbProps {
  items: BreadcrumbItem[];
  canNavigateUp?: boolean;
  onNavigateUp?: () => void;
}

const HeaderBreadcrumb = ({
  items,
  canNavigateUp,
  onNavigateUp,
}: HeaderBreadcrumbProps): JSX.Element => {
  return (
    <div className="flex items-center gap-1 text-xs text-muted-foreground">
      {onNavigateUp && (
        <button
          type="button"
          onClick={onNavigateUp}
          disabled={!canNavigateUp}
          aria-label="Back"
          className="flex h-6 w-6 items-center justify-center rounded-sm transition-colors hover:bg-muted/60 hover:text-foreground disabled:opacity-40"
        >
          <ChevronLeft className="h-3.5 w-3.5" />
        </button>
      )}
      <nav className="flex min-w-0 flex-wrap items-center gap-1">
        {items.map((item, index) => {
          const isLast = index === items.length - 1;
          return (
            <span
              key={`${item.label}-${item.to ?? index}`}
              className="flex min-w-0 items-center gap-1"
            >
              {item.to && !isLast ? (
                <Link
                  to={item.to}
                  className="truncate rounded-sm px-1 py-0.5 transition-colors hover:bg-muted/60 hover:text-foreground"
                >
                  {item.label}
                </Link>
              ) : (
                <span className={isLast ? "truncate text-foreground" : "truncate"}>
                  {item.label}
                </span>
              )}
              {!isLast && <ChevronRight className="h-3 w-3 flex-none opacity-50" />}
            </span>
          );
        })}
      </nav>
    </div>
  );
};

interface EntityHeaderProps {
  breadcrumbs?: BreadcrumbItem[];
  canNavigateUp?: boolean;
  onNavigateUp?: () => void;
  icon: ComponentType<{ className?: string }>;
  title: string;
  subtitle?: string;
  status?: string;
  actions?: ReactNode;
  metrics?: ReactNode;
}

export const EntityHeader = ({
  breadcrumbs,
  canNavigateUp,
  onNavigateUp,
  icon: Icon,
  title,
  subtitle,
  status,
  actions,
  metrics,
}: EntityHeaderProps): JSX.Element => {
  return (
    <section className="bg-background">
      <div className="px-6 pt-4 md:px-8">
        {breadcrumbs && breadcrumbs.length > 0 && (
          <HeaderBreadcrumb
            items={breadcrumbs}
            canNavigateUp={canNavigateUp}
            onNavigateUp={onNavigateUp}
          />
        )}

        <div className="mt-3 flex items-start justify-between gap-8 pb-6">
          <div className="flex min-w-0 flex-1 items-start gap-3">
            <div className="flex h-10 w-10 flex-none items-center justify-center rounded-md bg-muted/60">
              <Icon className="h-5 w-5 text-foreground" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <h2 className="truncate text-2xl font-semibold tracking-tight text-foreground">
                  {title}
                </h2>
                {status && <StatusBadge status={status} />}
              </div>
              {subtitle && (
                <p className="mt-1 max-w-2xl truncate text-sm text-muted-foreground">{subtitle}</p>
              )}
            </div>
          </div>

          {(actions || metrics) && (
            <div className="flex flex-none flex-col items-end gap-3">
              {actions && <div className="flex items-center gap-2">{actions}</div>}
              {metrics && (
                <div className="flex flex-wrap items-baseline justify-end gap-x-5 gap-y-1">
                  {metrics}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

interface KeyValueItem {
  label: string;
  value: ReactNode;
}

interface KeyValueGridProps {
  items: KeyValueItem[];
}

export const KeyValueGrid = ({ items }: KeyValueGridProps): JSX.Element => {
  return (
    <div className="grid gap-x-8 gap-y-4 md:grid-cols-2">
      {items.map((item) => (
        <div key={item.label} className="border-b border-border/60 pb-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            {item.label}
          </div>
          <div className="mt-1 text-sm text-foreground">{item.value}</div>
        </div>
      ))}
    </div>
  );
};
