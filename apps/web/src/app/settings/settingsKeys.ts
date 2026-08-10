/** TanStack Query keys for Settings panels. */
export const settingsKeys = {
  all: ["settings"] as const,
  remoteWorkspaces: () => [...settingsKeys.all, "remote-workspaces"] as const,
  computeTargets: () => [...settingsKeys.all, "compute-targets"] as const,
};
