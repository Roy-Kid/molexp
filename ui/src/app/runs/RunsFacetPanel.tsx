import { useMemo, useState } from "react";
import type { JSX } from "react";

import { cn } from "@/lib/utils";

import type { FacetCount, FacetSnapshot } from "./aggregates";
import {
  type ArrayFilterKey,
  hasActiveFilters,
  toggleArrayFilter,
  toggleQuickView,
} from "./filterParams";
import type { RunsQuickView, WorkspaceRunsFilters } from "./types";

interface RunsFacetPanelProps {
  facets: FacetSnapshot;
  filters: WorkspaceRunsFilters;
  onFiltersChange: (next: WorkspaceRunsFilters) => void;
}

const QUICK_VIEWS: Array<{ id: RunsQuickView; label: string; help: string }> = [
  { id: "active", label: "Active now", help: "running or pending" },
  { id: "failed24h", label: "Failed in 24h", help: "finished failed within 24h" },
  { id: "longRunning", label: "Long running", help: "running for >1h" },
];

const STATUS_LABELS: Record<string, string> = {
  running: "Running",
  pending: "Pending",
  succeeded: "Succeeded",
  failed: "Failed",
  cancelled: "Cancelled",
  skipped: "Skipped",
  timed_out: "Timed out",
  lost: "Lost",
  queued: "Queued",
  submitted: "Submitted",
};

const COLLAPSE_THRESHOLD = 8;

export const RunsFacetPanel = ({
  facets,
  filters,
  onFiltersChange,
}: RunsFacetPanelProps): JSX.Element => {
  const showExperiment = (filters.projectId?.length ?? 0) > 0;
  const showCluster = facets.cluster.length > 0;
  const active = hasActiveFilters(filters);

  return (
    <div className="space-y-4 px-1 pb-4">
      <FacetGroup title="Quick views">
        <div className="space-y-0.5">
          {QUICK_VIEWS.map((view) => {
            const count = facets.quickView[view.id];
            const checked = filters.quickView?.includes(view.id) ?? false;
            return (
              <FacetCheckboxRow
                key={view.id}
                label={view.label}
                title={view.help}
                count={count}
                checked={checked}
                onToggle={() => onFiltersChange(toggleQuickView(filters, view.id))}
              />
            );
          })}
        </div>
      </FacetGroup>

      <CheckboxFacetGroup
        title="Status"
        facetKey="status"
        options={facets.status}
        labels={STATUS_LABELS}
        filters={filters}
        onFiltersChange={onFiltersChange}
      />

      <CheckboxFacetGroup
        title="Backend"
        facetKey="backend"
        options={facets.backend}
        filters={filters}
        onFiltersChange={onFiltersChange}
      />

      {showCluster && (
        <CheckboxFacetGroup
          title="Cluster"
          facetKey="cluster"
          options={facets.cluster}
          filters={filters}
          onFiltersChange={onFiltersChange}
        />
      )}

      <CheckboxFacetGroup
        title="Project"
        facetKey="projectId"
        options={facets.projectId}
        filters={filters}
        onFiltersChange={onFiltersChange}
      />

      {showExperiment && (
        <CheckboxFacetGroup
          title="Experiment"
          facetKey="experimentId"
          options={facets.experimentId}
          filters={filters}
          onFiltersChange={onFiltersChange}
        />
      )}

      {active && (
        <button
          type="button"
          onClick={() => onFiltersChange({})}
          className="w-full rounded border border-border px-2 py-1.5 text-[11px] text-muted-foreground transition-colors hover:bg-muted/40 hover:text-foreground"
        >
          Reset filters
        </button>
      )}
    </div>
  );
};

interface FacetGroupProps {
  title: string;
  children: JSX.Element | JSX.Element[];
}

const FacetGroup = ({ title, children }: FacetGroupProps): JSX.Element => (
  <div>
    <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
      {title}
    </div>
    {children}
  </div>
);

interface FacetCheckboxRowProps {
  label: string;
  count: number;
  checked: boolean;
  onToggle: () => void;
  title?: string;
}

const FacetCheckboxRow = ({
  label,
  count,
  checked,
  onToggle,
  title,
}: FacetCheckboxRowProps): JSX.Element => {
  const dimmed = count === 0 && !checked;
  return (
    <label
      title={title}
      className={cn(
        "flex cursor-pointer items-center justify-between rounded px-1.5 py-1 text-xs transition-colors hover:bg-muted/40",
        dimmed && "cursor-default opacity-40 hover:bg-transparent",
      )}
    >
      <span className="flex min-w-0 items-center gap-2">
        <input
          type="checkbox"
          checked={checked}
          disabled={dimmed}
          onChange={onToggle}
          className="h-3 w-3 cursor-pointer rounded border border-border accent-primary"
        />
        <span className="truncate">{label}</span>
      </span>
      <span className="ml-2 shrink-0 tabular-nums text-[10px] text-muted-foreground">{count}</span>
    </label>
  );
};

interface CheckboxFacetGroupProps {
  title: string;
  facetKey: ArrayFilterKey;
  options: FacetCount[];
  labels?: Record<string, string>;
  filters: WorkspaceRunsFilters;
  onFiltersChange: (next: WorkspaceRunsFilters) => void;
}

const CheckboxFacetGroup = ({
  title,
  facetKey,
  options,
  labels,
  filters,
  onFiltersChange,
}: CheckboxFacetGroupProps): JSX.Element => {
  const [expanded, setExpanded] = useState(false);
  const selected = useMemo(() => new Set(filters[facetKey] ?? []), [filters, facetKey]);

  const merged = useMemo(() => {
    const map = new Map<string, FacetCount>();
    for (const option of options) map.set(option.value, option);
    for (const value of selected) {
      if (!map.has(value)) map.set(value, { value, label: labels?.[value] ?? value, count: 0 });
    }
    return Array.from(map.values()).sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return a.label.localeCompare(b.label);
    });
  }, [options, selected, labels]);

  if (merged.length === 0) {
    return (
      <FacetGroup title={title}>
        <div className="px-1.5 py-1 text-[11px] italic text-muted-foreground">no values</div>
      </FacetGroup>
    );
  }

  const isCollapsed = !expanded && merged.length > COLLAPSE_THRESHOLD;
  const visible = isCollapsed ? merged.slice(0, COLLAPSE_THRESHOLD) : merged;

  return (
    <FacetGroup title={title}>
      <div className="space-y-0.5">
        {visible.map((option) => (
          <FacetCheckboxRow
            key={option.value}
            label={labels?.[option.value] ?? option.label}
            count={option.count}
            checked={selected.has(option.value)}
            onToggle={() =>
              onFiltersChange(toggleArrayFilter(filters, facetKey, option.value))
            }
          />
        ))}
        {merged.length > COLLAPSE_THRESHOLD && (
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="ml-1 mt-0.5 text-[10px] uppercase tracking-wide text-muted-foreground hover:text-foreground"
          >
            {expanded ? "Show fewer" : `Show ${merged.length - COLLAPSE_THRESHOLD} more`}
          </button>
        )}
      </div>
    </FacetGroup>
  );
};
