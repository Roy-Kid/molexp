import { useCallback, useMemo, useState } from "react";

/**
 * Range/incremental multi-selection for the experiment run list. The repo has no
 * prior shift/ctrl selection helper, so the transition is captured in a pure
 * reducer (`nextSelection`) that the hook wraps — keeping the modifier logic
 * testable under the node environment, independent of React.
 */

export interface MultiSelectState {
  selected: Set<string>;
  /** Index of the last plain/ctrl click — origin for a subsequent shift range. */
  anchor: number | null;
}

export interface ClickModifiers {
  /** shiftKey — select the inclusive range from the anchor. */
  shift: boolean;
  /** metaKey || ctrlKey — toggle the single row, preserving the rest. */
  meta: boolean;
}

/**
 * Pure selection transition:
 *  - shift (with an anchor) → add the inclusive index range [anchor, click] to
 *    the current selection, anchor unchanged.
 *  - meta/ctrl → toggle just the clicked id, anchor moves to the click.
 *  - plain → select only the clicked id, anchor moves to the click.
 */
export const nextSelection = (
  state: MultiSelectState,
  clickIndex: number,
  orderedIds: string[],
  modifiers: ClickModifiers,
): MultiSelectState => {
  const id = orderedIds[clickIndex];

  if (modifiers.shift && state.anchor !== null) {
    const [lo, hi] =
      state.anchor <= clickIndex ? [state.anchor, clickIndex] : [clickIndex, state.anchor];
    const selected = new Set(state.selected);
    for (let i = lo; i <= hi; i += 1) {
      selected.add(orderedIds[i]);
    }
    return { selected, anchor: state.anchor };
  }

  if (modifiers.meta) {
    const selected = new Set(state.selected);
    if (selected.has(id)) {
      selected.delete(id);
    } else {
      selected.add(id);
    }
    return { selected, anchor: clickIndex };
  }

  return { selected: new Set([id]), anchor: clickIndex };
};

export interface UseRunMultiSelect {
  /** Whether multi-select mode is active (rows become selectable, nav suppressed). */
  enabled: boolean;
  selected: Set<string>;
  toggleMode: () => void;
  /** Apply a click at `index` with the given keyboard modifiers. */
  selectAt: (index: number, modifiers: ClickModifiers) => void;
  clear: () => void;
}

export const useRunMultiSelect = (orderedIds: string[]): UseRunMultiSelect => {
  const [enabled, setEnabled] = useState(false);
  const [state, setState] = useState<MultiSelectState>(() => ({
    selected: new Set<string>(),
    anchor: null,
  }));

  const selectAt = useCallback(
    (index: number, modifiers: ClickModifiers) => {
      setState((current) => nextSelection(current, index, orderedIds, modifiers));
    },
    [orderedIds],
  );

  const clear = useCallback(() => {
    setState({ selected: new Set<string>(), anchor: null });
  }, []);

  const toggleMode = useCallback(() => {
    setEnabled((value) => !value);
    setState({ selected: new Set<string>(), anchor: null });
  }, []);

  return useMemo(
    () => ({ enabled, selected: state.selected, toggleMode, selectAt, clear }),
    [enabled, state.selected, toggleMode, selectAt, clear],
  );
};
