/**
 * Pure helpers for the workspace Jobs table: duration, sort, pagination.
 * Kept free of React so sort/page rules are unit-testable.
 */

import type { WorkspaceRunRow } from "./types";

export type JobsSortKey =
  | "status"
  | "name"
  | "project"
  | "backend"
  | "attempts"
  | "duration"
  | "submitted";

export type SortDir = "asc" | "desc";

export interface JobsSort {
  key: JobsSortKey;
  dir: SortDir;
}

export const JOBS_SORT_KEYS: readonly JobsSortKey[] = [
  "status",
  "name",
  "project",
  "backend",
  "attempts",
  "duration",
  "submitted",
] as const;

export const DEFAULT_JOBS_SORT: JobsSort = { key: "submitted", dir: "desc" };
export const DEFAULT_PAGE_SIZE = 50;
export const PAGE_SIZE_OPTIONS = [25, 50, 100, 200] as const;

const SORT_KEY_SET: ReadonlySet<string> = new Set(JOBS_SORT_KEYS);

/** Earliest started execution → finished (or now). Seconds, or null if never started. */
export const computeRunDurationSeconds = (run: WorkspaceRunRow): number | null => {
  const start = run.executions
    .map((e) => (e.startedAt ? new Date(e.startedAt).getTime() : NaN))
    .filter((v) => !Number.isNaN(v))
    .sort((a, b) => a - b)[0];
  if (typeof start !== "number") return null;
  const end = run.finishedAt ? new Date(run.finishedAt).getTime() : Date.now();
  if (Number.isNaN(end)) return null;
  return Math.max(0, (end - start) / 1000);
};

const cmpString = (a: string, b: string): number =>
  a.localeCompare(b, undefined, { sensitivity: "base", numeric: true });

/** Compare numbers with nulls always last (regardless of sort dir). */
const cmpNullableNumber = (a: number | null, b: number | null, dir: SortDir): number => {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  const result = a - b;
  return dir === "desc" ? -result : result;
};

const sortValue = (run: WorkspaceRunRow, key: JobsSortKey): string | number | null => {
  switch (key) {
    case "status":
      return run.status;
    case "name":
      return run.name || run.id;
    case "project":
      return `${run.projectName}\0${run.experimentName}`;
    case "backend":
      return run.backend ?? "";
    case "attempts":
      return run.executionCount;
    case "duration":
      return computeRunDurationSeconds(run);
    case "submitted":
      return Date.parse(run.createdAt) || 0;
  }
};

export const compareJobs = (a: WorkspaceRunRow, b: WorkspaceRunRow, sort: JobsSort): number => {
  const av = sortValue(a, sort.key);
  const bv = sortValue(b, sort.key);
  let result: number;
  if (typeof av === "number" || typeof bv === "number" || av === null || bv === null) {
    // Duration (and similar) can be null; keep nulls last in both directions.
    if (sort.key === "duration") {
      result = cmpNullableNumber(
        typeof av === "number" ? av : null,
        typeof bv === "number" ? bv : null,
        sort.dir,
      );
    } else {
      const an = typeof av === "number" ? av : av === null ? 0 : Number(av);
      const bn = typeof bv === "number" ? bv : bv === null ? 0 : Number(bv);
      result = sort.dir === "desc" ? bn - an : an - bn;
    }
  } else {
    result = cmpString(String(av), String(bv));
    if (sort.dir === "desc") result = -result;
  }
  if (result === 0) {
    // Stable secondary key so equal rows don't thrash across pages.
    result = cmpString(a.id, b.id);
  }
  return result;
};

export const sortJobs = (rows: WorkspaceRunRow[], sort: JobsSort): WorkspaceRunRow[] => {
  if (rows.length <= 1) return rows;
  const copy = rows.slice();
  copy.sort((a, b) => compareJobs(a, b, sort));
  return copy;
};

export const parseJobsSort = (raw: string | null | undefined): JobsSort => {
  if (!raw) return DEFAULT_JOBS_SORT;
  const [keyPart, dirPart] = raw.split(":");
  const key = SORT_KEY_SET.has(keyPart) ? (keyPart as JobsSortKey) : DEFAULT_JOBS_SORT.key;
  const dir: SortDir = dirPart === "asc" || dirPart === "desc" ? dirPart : DEFAULT_JOBS_SORT.dir;
  return { key, dir };
};

export const formatJobsSort = (sort: JobsSort): string => `${sort.key}:${sort.dir}`;

export const parsePage = (raw: string | null | undefined): number => {
  if (!raw) return 1;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : 1;
};

export const parsePageSize = (raw: string | null | undefined): number => {
  if (!raw) return DEFAULT_PAGE_SIZE;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n)) return DEFAULT_PAGE_SIZE;
  return (PAGE_SIZE_OPTIONS as readonly number[]).includes(n) ? n : DEFAULT_PAGE_SIZE;
};

export interface PageSlice<T> {
  items: T[];
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
}

/** Clamp page into range and slice. Empty input → page 1, empty items. */
export const paginate = <T>(items: T[], page: number, pageSize: number): PageSlice<T> => {
  const totalItems = items.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize) || 1);
  const safePage = Math.min(Math.max(1, page), totalPages);
  const start = (safePage - 1) * pageSize;
  return {
    items: items.slice(start, start + pageSize),
    page: safePage,
    pageSize,
    totalItems,
    totalPages,
  };
};

/** Toggle sort: same key flips dir; new key starts desc for time-like, asc otherwise. */
export const nextJobsSort = (current: JobsSort, key: JobsSortKey): JobsSort => {
  if (current.key === key) {
    return { key, dir: current.dir === "asc" ? "desc" : "asc" };
  }
  const dir: SortDir =
    key === "submitted" || key === "duration" || key === "attempts" ? "desc" : "asc";
  return { key, dir };
};
