import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export interface DashboardLayoutState {
  order: string[];
  hidden: string[];
}

export interface UseDashboardLayoutResult<T extends string> {
  visibleIds: T[];
  hiddenIds: T[];
  reorder: (activeId: string, overId: string) => void;
  hide: (id: string) => void;
  restore: (id: string) => void;
  reset: () => void;
}

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string");

const readStored = (storageKey: string): DashboardLayoutState | null => {
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      isStringArray((parsed as { order?: unknown }).order) &&
      isStringArray((parsed as { hidden?: unknown }).hidden)
    ) {
      return parsed as DashboardLayoutState;
    }
  } catch {
    // ignore corrupt payloads — fall back to defaults
  }
  return null;
};

export const reconcileLayout = (
  stored: DashboardLayoutState | null,
  defaults: readonly string[],
): DashboardLayoutState => {
  const validIds = new Set<string>(defaults);
  const storedOrder = stored?.order.filter((id) => validIds.has(id)) ?? [];
  const seen = new Set(storedOrder);
  const merged = [...storedOrder, ...defaults.filter((id) => !seen.has(id))];
  const hidden = (stored?.hidden ?? []).filter((id) => validIds.has(id));
  return { order: merged, hidden };
};

export const moveItemBefore = <T>(
  list: readonly T[],
  activeId: T,
  overId: T,
): T[] => {
  if (activeId === overId) return [...list];
  const fromIdx = list.indexOf(activeId);
  const toIdx = list.indexOf(overId);
  if (fromIdx === -1 || toIdx === -1) return [...list];
  const next = [...list];
  next.splice(fromIdx, 1);
  next.splice(toIdx, 0, activeId);
  return next;
};

export const useDashboardLayout = <T extends string>(
  storageKey: string,
  defaultIds: readonly T[],
): UseDashboardLayoutResult<T> => {
  const defaultsRef = useRef(defaultIds);
  defaultsRef.current = defaultIds;

  const [state, setState] = useState<DashboardLayoutState>(() =>
    reconcileLayout(readStored(storageKey), defaultIds),
  );

  const didMount = useRef(false);
  useEffect(() => {
    if (!didMount.current) {
      didMount.current = true;
      return;
    }
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(state));
    } catch {
      // quota or privacy mode — silently ignore
    }
  }, [storageKey, state]);

  const reorder = useCallback((activeId: string, overId: string) => {
    setState((prev) => ({
      ...prev,
      order: moveItemBefore(prev.order, activeId, overId),
    }));
  }, []);

  const hide = useCallback((id: string) => {
    setState((prev) =>
      prev.hidden.includes(id) ? prev : { ...prev, hidden: [...prev.hidden, id] },
    );
  }, []);

  const restore = useCallback((id: string) => {
    setState((prev) => ({ ...prev, hidden: prev.hidden.filter((x) => x !== id) }));
  }, []);

  const reset = useCallback(() => {
    setState({ order: [...defaultsRef.current], hidden: [] });
  }, []);

  const hiddenSet = useMemo(() => new Set(state.hidden), [state.hidden]);
  const visibleIds = useMemo(
    () => state.order.filter((id) => !hiddenSet.has(id)) as T[],
    [state.order, hiddenSet],
  );

  return {
    visibleIds,
    hiddenIds: state.hidden as T[],
    reorder,
    hide,
    restore,
    reset,
  };
};
