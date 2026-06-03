import { createContext, useContext } from "react";

/**
 * A workflow-graph node "pinned" to the right inspector. This is deliberately
 * *not* part of the URL `Selection` / navigation state: clicking a node opens
 * the inspector in-place over the current run page rather than navigating to a
 * standalone task page.
 */
export interface InspectedTask {
  taskId: string;
  runId: string;
}

export interface InspectedTaskContextValue {
  /** The node currently pinned to the right inspector, or `null`. */
  inspectedTask: InspectedTask | null;
  /** Pin a node to the right inspector and open the panel. */
  inspectTask: (taskId: string, runId: string) => void;
  /** Unpin — the inspector reverts to the current page's own object. */
  clearInspectedTask: () => void;
}

const noop = (): void => {};

export const InspectedTaskContext = createContext<InspectedTaskContextValue>({
  inspectedTask: null,
  inspectTask: noop,
  clearInspectedTask: noop,
});

export const useInspectedTask = (): InspectedTaskContextValue => useContext(InspectedTaskContext);
