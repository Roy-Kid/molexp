import { Settings2 } from "lucide-react";
import type { ReactNode } from "react";
import { EntityPage } from "@/app/components/entity";

import { ComputeTargetsPanel } from "./ComputeTargetsPanel";
import { RemoteWorkspacesPanel } from "./RemoteWorkspacesPanel";

const TAB_REMOTE_WORKSPACES = "remote-workspaces" as const;
const TAB_COMPUTE_TARGETS = "compute-targets" as const;

const SettingsCanvas = ({ children }: { children: ReactNode }): JSX.Element => (
  <div className="flex-1 overflow-auto">
    <div className="mx-auto w-full max-w-5xl px-4 py-5 sm:px-6 sm:py-6">{children}</div>
  </div>
);

export function SettingsPage(): JSX.Element {
  const tabs = [
    {
      value: TAB_REMOTE_WORKSPACES,
      label: "Remote workspaces",
      content: (
        <SettingsCanvas>
          <RemoteWorkspacesPanel />
        </SettingsCanvas>
      ),
    },
    {
      value: TAB_COMPUTE_TARGETS,
      label: "Compute targets",
      content: (
        <SettingsCanvas>
          <ComputeTargetsPanel />
        </SettingsCanvas>
      ),
    },
  ];

  return (
    <EntityPage
      icon={Settings2}
      title="Workspace settings"
      subtitle="Connections and execution destinations for this workspace"
      defaultTab={TAB_REMOTE_WORKSPACES}
      tabs={tabs}
    />
  );
}
