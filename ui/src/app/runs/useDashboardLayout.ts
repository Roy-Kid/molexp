import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export type DropPosition = "left" | "right" | "top" | "bottom";

export interface DashboardRow<T extends string = string> {
  id: string;
  panels: T[];
}

export interface DashboardLayoutState {
  rows: DashboardRow[];
  hidden: string[];
}

export interface UseDashboardLayoutResult<T extends string> {
  rows: DashboardRow<T>[];
  hiddenIds: T[];
  visibleIds: T[];
  reorder: (activeId: string, overId: string, position: DropPosition) => void;
  hide: (id: string) => void;
  restore: (id: string) => void;
  toggleVisibility: (id: string) => void;
  reset: () => void;
}

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string");

const isRowsArray = (value: unknown): value is DashboardRow[] =>
  Array.isArray(value) &&
  value.every(
    (item) =>
      typeof item === "object" &&
      item !== null &&
      typeof (item as DashboardRow).id === "string" &&
      isStringArray((item as DashboardRow).panels),
  );

export const generateRowId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `row_${Math.random().toString(36).slice(2, 10)}_${Date.now().toString(36)}`;
};

const readStored = (storageKey: string): DashboardLayoutState | null => {
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      isRowsArray((parsed as { rows?: unknown }).rows) &&
      isStringArray((parsed as { hidden?: unknown }).hidden)
    ) {
      return parsed as DashboardLayoutState;
    }
  } catch {
    // ignore corrupt payloads — fall back to defaults
  }
  return null;
};

const buildDefaults = (defaults: readonly string[]): DashboardLayoutState => ({
  rows: defaults.map((id) => ({ id: generateRowId(), panels: [id] })),
  hidden: [],
});

export const reconcileLayout = (
  stored: DashboardLayoutState | null,
  defaults: readonly string[],
): DashboardLayoutState => {
  if (!stored) return buildDefaults(defaults);

  const validIds = new Set(defaults);
  const seen = new Set<string>();
  const rows: DashboardRow[] = [];
  for (const row of stored.rows) {
    const panels = row.panels.filter((id) => validIds.has(id) && !seen.has(id));
    panels.forEach((id) => seen.add(id));
    if (panels.length > 0) {
      rows.push({ id: row.id, panels });
    }
  }

  const hidden = stored.hidden.filter((id) => validIds.has(id));
  const hiddenSet = new Set(hidden);

  for (const id of defaults) {
    if (!seen.has(id) && !hiddenSet.has(id)) {
      rows.push({ id: generateRowId(), panels: [id] });
      seen.add(id);
    }
  }

  return { rows, hidden };
};

const removePanel = (rows: DashboardRow[], panelId: string): DashboardRow[] =>
  rows
    .map((row) => ({ ...row, panels: row.panels.filter((p) => p !== panelId) }))
    .filter((row) => row.panels.length > 0);

const findRowIndex = (rows: DashboardRow[], panelId: string): number =>
  rows.findIndex((row) => row.panels.includes(panelId));

export const applyReorder = (
  rows: DashboardRow[],
  activeId: string,
  overId: string,
  position: DropPosition,
): DashboardRow[] => {
  if (activeId === overId) return rows;

  const cleaned = removePanel(rows, activeId);
  const targetRowIdx = findRowIndex(cleaned, overId);
  if (targetRowIdx === -1) return rows;

  const targetRow = cleaned[targetRowIdx];
  const targetPanelIdx = targetRow.panels.indexOf(overId);

  if (position === "left" || position === "right") {
    const insertAt = position === "left" ? targetPanelIdx : targetPanelIdx + 1;
    const nextRow: DashboardRow = {
      ...targetRow,
      panels: [
        ...targetRow.panels.slice(0, insertAt),
        activeId,
        ...targetRow.panels.slice(insertAt),
      ],
    };
    return cleaned.map((row, idx) => (idx === targetRowIdx ? nextRow : row));
  }

  const newRow: DashboardRow = { id: generateRowId(), panels: [activeId] };
  const insertRowAt = position === "top" ? targetRowIdx : targetRowIdx + 1;
  return [
    ...cleaned.slice(0, insertRowAt),
    newRow,
    ...cleaned.slice(insertRowAt),
  ];
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

  const reorder = useCallback(
    (activeId: string, overId: string, position: DropPosition) => {
      setState((prev) => ({
        ...prev,
        rows: applyReorder(prev.rows, activeId, overId, position),
      }));
    },
    [],
  );

  const hide = useCallback((id: string) => {
    setState((prev) => {
      if (prev.hidden.includes(id)) return prev;
      return {
        rows: removePanel(prev.rows, id),
        hidden: [...prev.hidden, id],
      };
    });
  }, []);

  const restore = useCallback((id: string) => {
    setState((prev) => {
      if (!prev.hidden.includes(id)) return prev;
      return {
        rows: [...prev.rows, { id: generateRowId(), panels: [id] }],
        hidden: prev.hidden.filter((x) => x !== id),
      };
    });
  }, []);

  const toggleVisibility = useCallback((id: string) => {
    setState((prev) => {
      if (prev.hidden.includes(id)) {
        return {
          rows: [...prev.rows, { id: generateRowId(), panels: [id] }],
          hidden: prev.hidden.filter((x) => x !== id),
        };
      }
      return {
        rows: removePanel(prev.rows, id),
        hidden: [...prev.hidden, id],
      };
    });
  }, []);

  const reset = useCallback(() => {
    setState(buildDefaults(defaultsRef.current));
  }, []);

  const visibleIds = useMemo(
    () => state.rows.flatMap((row) => row.panels) as T[],
    [state.rows],
  );

  return {
    rows: state.rows as DashboardRow<T>[],
    hiddenIds: state.hidden as T[],
    visibleIds,
    reorder,
    hide,
    restore,
    toggleVisibility,
    reset,
  };
};
