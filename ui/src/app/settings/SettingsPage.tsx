/**
 * Settings page — two-tab layout: Remote workspaces (default) and
 * Compute targets.  Future sections (profiles, MCP integrations) slot
 * in as additional tabs.
 */

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { ComputeTargetsPanel } from "./ComputeTargetsPanel";
import { RemoteWorkspacesPanel } from "./RemoteWorkspacesPanel";

const TAB_REMOTE_WORKSPACES = "remote-workspaces" as const;
const TAB_COMPUTE_TARGETS = "compute-targets" as const;

export function SettingsPage(): JSX.Element {
  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-5xl space-y-6 px-6 py-6">
        <header className="space-y-1">
          <h2 className="text-base font-semibold text-foreground">Workspace settings</h2>
          <p className="text-sm text-muted-foreground">
            Remote workspaces and compute targets are stored at{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-xs">~/.molexp/</code> (remote-root
            descriptors) and{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-xs">workspace.json</code> (compute
            targets) respectively.
          </p>
        </header>
        <Tabs defaultValue={TAB_REMOTE_WORKSPACES} className="space-y-4">
          <TabsList>
            <TabsTrigger value={TAB_REMOTE_WORKSPACES}>Remote workspaces</TabsTrigger>
            <TabsTrigger value={TAB_COMPUTE_TARGETS}>Compute targets</TabsTrigger>
          </TabsList>
          <TabsContent value={TAB_REMOTE_WORKSPACES}>
            <RemoteWorkspacesPanel />
          </TabsContent>
          <TabsContent value={TAB_COMPUTE_TARGETS}>
            <ComputeTargetsPanel />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
