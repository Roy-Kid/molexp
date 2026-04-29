/**
 * Center-panel content for the Settings view (workspace-level configuration).
 * Currently hosts the ComputeTargetsPanel; future sections (profiles, MCP
 * integrations) slot in here as additional <section> blocks.
 */

import { ComputeTargetsPanel } from "./ComputeTargetsPanel";

export function SettingsPage(): JSX.Element {
  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-5xl space-y-8 px-6 py-6">
        <header className="space-y-1">
          <h2 className="text-base font-semibold text-foreground">Workspace settings</h2>
          <p className="text-sm text-muted-foreground">
            Configuration scoped to this workspace. Changes are persisted to{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-xs">workspace.json</code>.
          </p>
        </header>
        <ComputeTargetsPanel />
      </div>
    </div>
  );
}
