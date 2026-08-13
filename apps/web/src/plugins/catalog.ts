/**
 * UI plugin catalog — metadata for the settings toggle UI.
 *
 * Populated when each plugin is registered through the loader. Core and
 * third-party bundles both land here so the user can enable/disable panel
 * contributions without a page reload.
 */

export interface PluginCatalogEntry {
  id: string;
  name: string;
  description?: string;
  /**
   * When false the plugin always contributes and is hidden from the toggle
   * list (e.g. `core` shell renderers). Defaults to true.
   */
  userToggleable: boolean;
}

const catalog = new Map<string, PluginCatalogEntry>();

export const registerPluginCatalogEntry = (entry: PluginCatalogEntry): void => {
  catalog.set(entry.id, {
    id: entry.id,
    name: entry.name,
    description: entry.description,
    userToggleable: entry.userToggleable,
  });
};

export const listPluginCatalog = (): PluginCatalogEntry[] => {
  return Array.from(catalog.values()).sort((a, b) => a.name.localeCompare(b.name));
};

export const getPluginCatalogEntry = (id: string): PluginCatalogEntry | undefined => {
  return catalog.get(id);
};

/** Toggleable plugins only — what the Settings UI lists. */
export const listToggleablePlugins = (): PluginCatalogEntry[] => {
  return listPluginCatalog().filter((entry) => entry.userToggleable);
};

export const resetPluginCatalogForTests = (): void => {
  catalog.clear();
};
