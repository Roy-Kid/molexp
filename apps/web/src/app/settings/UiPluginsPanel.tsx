/**
 * Settings → UI plugins — user toggles for panel contribution plugins.
 *
 * Each toggleable plugin (workflow, molplot, molvis, molq, …) can be turned
 * off so its center/right/tab contributions stop resolving. Core is always
 * on and never listed. Preferences persist in localStorage.
 */

import { Puzzle } from "lucide-react";
import type { JSX } from "react";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { listToggleablePlugins } from "@/plugins/catalog";
import {
  isPluginEnabled,
  setPluginEnabled,
  usePluginPreferencesGeneration,
} from "@/plugins/preferences";

export function UiPluginsPanel(): JSX.Element {
  usePluginPreferencesGeneration();
  const plugins = listToggleablePlugins();

  if (plugins.length === 0) {
    return (
      <p className="text-body text-muted-foreground">
        No toggleable UI plugins are registered yet.
      </p>
    );
  }

  return (
    <ul className="divide-y divide-border rounded-panel border border-border">
      {plugins.map((plugin) => {
        const enabled = isPluginEnabled(plugin.id);
        const switchId = `ui-plugin-${plugin.id}`;
        return (
          <li key={plugin.id} className="flex items-start gap-3 px-3 py-3 sm:items-center sm:px-4">
            <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-control bg-surface-subtle text-muted-foreground">
              <Puzzle className="size-3.5" aria-hidden />
            </div>
            <div className="min-w-0 flex-1 space-y-0.5">
              <Label
                htmlFor={switchId}
                className="cursor-pointer text-body font-medium text-foreground"
              >
                {plugin.name}
              </Label>
              {plugin.description ? (
                <p className="text-label text-muted-foreground">{plugin.description}</p>
              ) : null}
            </div>
            <Switch
              id={switchId}
              checked={enabled}
              onCheckedChange={(checked) => setPluginEnabled(plugin.id, checked)}
              aria-label={`${enabled ? "Disable" : "Enable"} ${plugin.name}`}
            />
          </li>
        );
      })}
    </ul>
  );
}
