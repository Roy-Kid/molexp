/**
 * Per-plugin enable/disable preferences for panel UI plugins.
 *
 * Stored in localStorage under a stable key. Missing keys default to
 * **enabled**. Non-toggleable plugins (e.g. `core`) always report enabled.
 *
 * Preference changes notify subscribers so React surfaces re-resolve
 * contributions without a full reload.
 */

import { useSyncExternalStore } from "react";

const STORAGE_KEY = "molexp.ui-plugins.enabled";

export type PluginEnabledMap = Record<string, boolean>;

let enabledMap: PluginEnabledMap = loadFromStorage();
let generation = 0;
const subscribers = new Set<() => void>();

function loadFromStorage(): PluginEnabledMap {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    const out: PluginEnabledMap = {};
    for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof value === "boolean") {
        out[key] = value;
      }
    }
    return out;
  } catch {
    return {};
  }
}

function persist(map: PluginEnabledMap): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch {
    // Quota / private mode — keep in-memory state only.
  }
}

const notify = (): void => {
  generation += 1;
  for (const fn of subscribers) {
    fn();
  }
};

/** Whether a plugin is currently enabled (missing key → true). */
export const isPluginEnabled = (pluginId: string): boolean => {
  const value = enabledMap[pluginId];
  return value !== false;
};

/** Snapshot of the enabled map (copy). */
export const getPluginEnabledMap = (): PluginEnabledMap => ({ ...enabledMap });

export const getPluginPreferencesGeneration = (): number => generation;

/**
 * Set a plugin's enabled flag. No-op when the value is unchanged.
 * Does not enforce `userToggleable` — callers (settings UI) should.
 */
export const setPluginEnabled = (pluginId: string, enabled: boolean): void => {
  if (enabledMap[pluginId] === enabled) {
    // Still treat explicit set as sticky even when default is true.
    if (pluginId in enabledMap) {
      return;
    }
  }
  if (enabled) {
    // Store true so users can re-enable after disable; keep the key sticky.
    enabledMap = { ...enabledMap, [pluginId]: true };
  } else {
    enabledMap = { ...enabledMap, [pluginId]: false };
  }
  persist(enabledMap);
  notify();
};

/** Test / boot hook: replace the whole map. */
export const replacePluginEnabledMap = (map: PluginEnabledMap): void => {
  enabledMap = { ...map };
  persist(enabledMap);
  notify();
};

/** Test hook: clear preferences (all plugins default-on). */
export const resetPluginPreferencesForTests = (): void => {
  enabledMap = {};
  if (typeof window !== "undefined") {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  }
  notify();
};

const subscribe = (fn: () => void): (() => void) => {
  subscribers.add(fn);
  return () => {
    subscribers.delete(fn);
  };
};

/** Re-render when any plugin enable flag changes. */
export const usePluginPreferencesGeneration = (): number =>
  useSyncExternalStore(subscribe, getPluginPreferencesGeneration, getPluginPreferencesGeneration);

/** Convenience: enabled flag for one plugin, reactive. */
export const usePluginEnabled = (pluginId: string): boolean => {
  usePluginPreferencesGeneration();
  return isPluginEnabled(pluginId);
};
