import { resetContributionRuntimeForTests } from "@/plugins/contribution-runtime";
import corePlugin from "@/plugins/core";
import type { UiPluginModule } from "@/plugins/types";

type LazyPluginLoader = () => Promise<{ default: UiPluginModule }>;

const lazyPluginLoaders: Record<string, LazyPluginLoader> = {
  metrics: () => import("@/plugins/metrics"),
  molvis: () => import("@/plugins/molvis"),
  molq: () => import("@/plugins/molq"),
};

const installed = new Set<string>();
const lazyPromises = new Map<string, Promise<void>>();

const registerPluginInstance = (plugin: UiPluginModule): void => {
  if (installed.has(plugin.id)) {
    return;
  }
  installed.add(plugin.id);
  const result = plugin.register();
  if (result instanceof Promise) {
    result.catch((error) => {
      console.warn(`[plugins] "${plugin.id}" register() rejected:`, error);
    });
  }
};

const loadLazyPlugin = (key: string): Promise<void> => {
  const cached = lazyPromises.get(key);
  if (cached) {
    return cached;
  }

  const loader = lazyPluginLoaders[key];
  if (!loader) {
    return Promise.resolve();
  }

  const promise = loader()
    .then((mod) => registerPluginInstance(mod.default))
    .catch((error) => {
      console.warn(`[plugins] Failed to load lazy plugin "${key}":`, error);
      lazyPromises.delete(key);
    });

  lazyPromises.set(key, promise);
  return promise;
};

registerPluginInstance(corePlugin);

const scheduleBackgroundLoad = (): void => {
  for (const key of Object.keys(lazyPluginLoaders)) {
    void loadLazyPlugin(key);
  }
};

if (typeof window !== "undefined") {
  const idle = (
    window as Window & {
      requestIdleCallback?: (cb: () => void) => void;
    }
  ).requestIdleCallback;
  if (idle) {
    idle(scheduleBackgroundLoad);
  } else {
    setTimeout(scheduleBackgroundLoad, 0);
  }
}

export const ensureLazyPlugin = (key: keyof typeof lazyPluginLoaders): Promise<void> => {
  return loadLazyPlugin(key);
};

export const resetUiPluginsForTests = (): void => {
  installed.clear();
  lazyPromises.clear();
  resetContributionRuntimeForTests();
};
