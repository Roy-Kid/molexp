/**
 * Workspace settings — molvis-style left nav + scroll sections.
 *
 * Sections: Remote workspaces, Compute targets, Users (admin when auth on).
 * Chrome is domain-free (`components/settings`); panels keep their own data.
 */

import { Cpu, Users, Wifi } from "lucide-react";
import { useMemo } from "react";
import { useAuth } from "@/app/auth";
import { SettingsSection, SettingsShell, type SettingsNavEntry } from "@/components/settings";
import { ComputeTargetsPanel } from "./ComputeTargetsPanel";
import { RemoteWorkspacesPanel } from "./RemoteWorkspacesPanel";
import { UsersSection } from "./UsersSection";

export function SettingsPage(): JSX.Element {
  const { enabled, user } = useAuth();
  const showUsers = enabled && user?.role === "admin";

  const entries = useMemo<SettingsNavEntry[]>(
    () => [
      {
        id: "remote-workspaces",
        label: "Remote workspaces",
        group: "workspace",
        groupLabel: "Workspace",
        icon: <Wifi className="size-3.5" aria-hidden />,
        content: (
          <SettingsSection
            id="remote-workspaces"
            title="Remote workspaces"
            description="SSH-backed roots the server can open as the active workspace."
          >
            <RemoteWorkspacesPanel />
          </SettingsSection>
        ),
      },
      {
        id: "compute-targets",
        label: "Compute targets",
        group: "workspace",
        groupLabel: "Workspace",
        icon: <Cpu className="size-3.5" aria-hidden />,
        content: (
          <SettingsSection
            id="compute-targets"
            title="Compute targets"
            description="Named execution backends for runs (local / remote schedulers)."
          >
            <ComputeTargetsPanel />
          </SettingsSection>
        ),
      },
      {
        id: "users",
        label: "Users",
        group: "access",
        groupLabel: "Access",
        icon: <Users className="size-3.5" aria-hidden />,
        hidden: !showUsers,
        content: <UsersSection sectionId="users" />,
      },
    ],
    [showUsers],
  );

  return (
    <div className="flex h-full min-h-0 flex-col p-3 sm:p-4">
      <SettingsShell
        title="Workspace settings"
        entries={entries}
        defaultId="remote-workspaces"
        className="h-full"
      />
    </div>
  );
}
