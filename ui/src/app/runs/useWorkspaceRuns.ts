import { useSyncExternalStore } from "react";

import { workspaceRunsApi } from "./api";
import type { WorkspaceRunRow, WorkspaceRunsResponse, WorkspaceRunsStats } from "./types";

interface UseWorkspaceRunsResult {
  rows: WorkspaceRunRow[];
  stats: WorkspaceRunsStats;
  total: number;
  truncated: boolean;
  loading: boolean;
  error: string | null;
  lastSyncedAt: Date | null;
  refresh: () => void;
}

interface StoreSnapshot {
  rows: WorkspaceRunRow[];
  stats: WorkspaceRunsStats;
  total: number;
  truncated: boolean;
  loading: boolean;
  error: string | null;
  lastSyncedAt: Date | null;
}

const POLL_INTERVAL_MS = 3_000;
const FETCH_LIMIT = 1000;

const EMPTY_STATS: WorkspaceRunsStats = {
  total: 0,
  running: 0,
  pending: 0,
  failed: 0,
  succeeded: 0,
};

const EMPTY_SNAPSHOT: StoreSnapshot = {
  rows: [],
  stats: EMPTY_STATS,
  total: 0,
  truncated: false,
  loading: false,
  error: null,
  lastSyncedAt: null,
};

// Module-level singleton: one fetch loop shared across all hook consumers
// (LeftPanel facet counts + RunsPage dashboard) so the API is hit once per
// poll instead of once per mounted component.
let snapshot: StoreSnapshot = EMPTY_SNAPSHOT;
let lastResponseSignature: string | null = null;
const subscribers = new Set<() => void>();
let intervalId: ReturnType<typeof setInterval> | null = null;
let inflight: Promise<void> | null = null;

const notify = (): void => {
  for (const fn of subscribers) fn();
};

const sameResponse = (response: WorkspaceRunsResponse): boolean => {
  const next = JSON.stringify(response);
  if (next === lastResponseSignature) return true;
  lastResponseSignature = next;
  return false;
};

const fetchOnce = async (silent: boolean): Promise<void> => {
  if (!silent && !snapshot.loading) {
    snapshot = { ...snapshot, loading: true };
    notify();
  }
  try {
    const response = await workspaceRunsApi.listRuns({ limit: FETCH_LIMIT });
    if (sameResponse(response)) {
      // Preserve row/stats array refs so downstream useMemo deps stay stable
      // (Plotly charts won't re-render). Only loading/error need to clear.
      if (snapshot.loading || snapshot.error !== null || snapshot.lastSyncedAt === null) {
        snapshot = {
          ...snapshot,
          loading: false,
          error: null,
          lastSyncedAt: snapshot.lastSyncedAt ?? new Date(),
        };
        notify();
      }
      return;
    }
    snapshot = {
      rows: response.runs,
      stats: response.stats,
      total: response.total,
      truncated: response.truncated,
      loading: false,
      error: null,
      lastSyncedAt: new Date(),
    };
    notify();
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    snapshot = { ...snapshot, loading: false, error: message };
    notify();
  }
};

const triggerFetch = (silent: boolean): void => {
  if (inflight) return;
  inflight = fetchOnce(silent).finally(() => {
    inflight = null;
  });
};

const subscribe = (fn: () => void): (() => void) => {
  if (subscribers.size === 0) {
    triggerFetch(false);
    intervalId = setInterval(() => triggerFetch(true), POLL_INTERVAL_MS);
  }
  subscribers.add(fn);
  return () => {
    subscribers.delete(fn);
    if (subscribers.size === 0 && intervalId !== null) {
      clearInterval(intervalId);
      intervalId = null;
    }
  };
};

const getSnapshot = (): StoreSnapshot => snapshot;

// No-op subscribe used when the caller asks us to stand down (e.g. when the
// active left-panel view isn't "runs"). Keeps the Rules of Hooks intact while
// letting subscriber count fall to zero so the poll loop stops.
const noopSubscribe = (): (() => void) => (): void => undefined;

export interface UseWorkspaceRunsOptions {
  /** When false, the hook returns the cached snapshot but does not subscribe
   *  or trigger fetches. Use to gate polling by active view. Defaults to true. */
  enabled?: boolean;
}

export const useWorkspaceRuns = (options: UseWorkspaceRunsOptions = {}): UseWorkspaceRunsResult => {
  const { enabled = true } = options;
  const data = useSyncExternalStore(enabled ? subscribe : noopSubscribe, getSnapshot, getSnapshot);
  return {
    ...data,
    refresh: () => triggerFetch(false),
  };
};
