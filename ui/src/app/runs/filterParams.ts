import type { RunsQuickView, WorkspaceRunsFilters } from "./types";

const ARRAY_KEYS = ["projectId", "experimentId", "backend", "cluster", "status"] as const;
type ArrayKey = (typeof ARRAY_KEYS)[number];

const QUICK_VIEW_VALUES: ReadonlySet<RunsQuickView> = new Set([
  "active",
  "failed24h",
  "longRunning",
]);

const splitCsv = (value: string): string[] =>
  value
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);

export const parseFilterParams = (params: URLSearchParams): WorkspaceRunsFilters => {
  const out: WorkspaceRunsFilters = {};
  for (const key of ARRAY_KEYS) {
    const raw = params.get(key);
    if (!raw) continue;
    const values = splitCsv(raw);
    if (values.length > 0) out[key] = values;
  }
  const quickRaw = params.get("quickView");
  if (quickRaw) {
    const quickValues = splitCsv(quickRaw).filter((value): value is RunsQuickView =>
      QUICK_VIEW_VALUES.has(value as RunsQuickView),
    );
    if (quickValues.length > 0) out.quickView = quickValues;
  }
  const limit = params.get("limit");
  if (limit) {
    const parsed = Number.parseInt(limit, 10);
    if (Number.isFinite(parsed) && parsed > 0) out.limit = parsed;
  }
  return out;
};

export const writeFilterParams = (
  prev: URLSearchParams,
  next: WorkspaceRunsFilters,
): URLSearchParams => {
  const merged = new URLSearchParams(prev);
  for (const key of ARRAY_KEYS) {
    const values = next[key];
    if (values && values.length > 0) merged.set(key, values.join(","));
    else merged.delete(key);
  }
  if (next.quickView && next.quickView.length > 0) {
    merged.set("quickView", next.quickView.join(","));
  } else {
    merged.delete("quickView");
  }
  if (next.limit !== undefined) merged.set("limit", String(next.limit));
  else merged.delete("limit");
  return merged;
};

export const toggleArrayFilter = (
  filters: WorkspaceRunsFilters,
  key: ArrayKey,
  value: string,
): WorkspaceRunsFilters => {
  const current = filters[key] ?? [];
  const exists = current.includes(value);
  const nextValues = exists ? current.filter((v) => v !== value) : [...current, value];
  const next: WorkspaceRunsFilters = { ...filters };
  if (nextValues.length > 0) next[key] = nextValues;
  else delete next[key];
  if (key === "projectId") delete next.experimentId;
  return next;
};

export const toggleQuickView = (
  filters: WorkspaceRunsFilters,
  view: RunsQuickView,
): WorkspaceRunsFilters => {
  const current = filters.quickView ?? [];
  const exists = current.includes(view);
  const nextValues = exists ? current.filter((v) => v !== view) : [...current, view];
  const next: WorkspaceRunsFilters = { ...filters };
  if (nextValues.length > 0) next.quickView = nextValues;
  else delete next.quickView;
  return next;
};

export const hasActiveFilters = (filters: WorkspaceRunsFilters): boolean => {
  for (const key of ARRAY_KEYS) {
    if ((filters[key]?.length ?? 0) > 0) return true;
  }
  return (filters.quickView?.length ?? 0) > 0;
};

export const ARRAY_FILTER_KEYS = ARRAY_KEYS;
export type ArrayFilterKey = ArrayKey;
