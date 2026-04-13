import { resetContributionRuntimeForTests } from "@/plugins/contribution-runtime";
import type { PluginManifest, UiPluginModule } from "@/plugins/types";

const builtinPluginLoaders: Record<string, () => Promise<{ default: UiPluginModule }>> = {
  core: () => import("@/plugins/core"),
  molq: () => import("@/plugins/molq"),
};

const installedPlugins = new Set<string>();
let initializationPromise: Promise<void> | null = null;

const loadBuiltinPlugin = async (moduleId: string): Promise<void> => {
  if (installedPlugins.has(moduleId)) {
    return;
  }

  const loader = builtinPluginLoaders[moduleId];
  if (!loader) {
    console.warn(`[plugins] No builtin module registered for "${moduleId}".`);
    return;
  }

  const module = await loader();
  await module.default.register();
  installedPlugins.add(moduleId);
  installedPlugins.add(module.default.id);
};

const fetchPluginManifests = async (): Promise<PluginManifest[]> => {
  const response = await fetch("/api/plugins");
  if (!response.ok) {
    throw new Error(`Plugin manifest request failed: ${response.status}`);
  }

  const payload = (await response.json()) as { plugins?: PluginManifest[] };
  return payload.plugins ?? [];
};

export const initializeUiPlugins = async (): Promise<void> => {
  if (initializationPromise) {
    return initializationPromise;
  }

  initializationPromise = (async () => {
    await loadBuiltinPlugin("core");

    try {
      const manifests = await fetchPluginManifests();
      for (const manifest of manifests) {
        const moduleId = manifest.uiModule ?? manifest.id;
        if (moduleId === "core") {
          continue;
        }
        await loadBuiltinPlugin(moduleId);
      }
    } catch (error) {
      console.warn("[plugins] Failed to load plugin manifests. Continuing with core UI only.", error);
    }
  })();

  return initializationPromise;
};

export const resetUiPluginsForTests = (): void => {
  installedPlugins.clear();
  initializationPromise = null;
  resetContributionRuntimeForTests();
};
