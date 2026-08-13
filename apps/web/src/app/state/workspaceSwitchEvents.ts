/**
 * Workspace-switch event emitter — mirrors mcpEvents.ts.
 *
 * On a workspace switch, any module that holds a long-lived
 * workspace-bound resource (open EventSource, file watcher, etc.) should
 * register a listener via `onWorkspaceSwitching` and tear down its
 * resource synchronously.  setActiveWorkspace fires the event *after*
 * the server-side switch returns but *before* it invalidates client
 * caches, so subscribers get a clean handoff.
 */

const EVENT_NAME = "workspace-switching" as const;

export interface WorkspaceSwitchingDetail {
  /** Name of the workspace-target now active, or null if cleared / unknown. */
  activeDescriptor: string | null;
}

export function emitWorkspaceSwitching(detail: WorkspaceSwitchingDetail): void {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(new CustomEvent(EVENT_NAME, { detail }));
}

export function onWorkspaceSwitching(
  handler: (detail: WorkspaceSwitchingDetail) => void,
): () => void {
  if (typeof window === "undefined") {
    return () => {};
  }
  const wrapped = (ev: Event): void => {
    handler((ev as CustomEvent<WorkspaceSwitchingDetail>).detail);
  };
  window.addEventListener(EVENT_NAME, wrapped);
  return () => window.removeEventListener(EVENT_NAME, wrapped);
}
