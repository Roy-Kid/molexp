/**
 * Plugin runtime bootstrap.
 *
 * Exposes `bootPlugins()` which `index.tsx` calls **after**
 * `enableMocking()` resolves, so the MSW service worker (in
 * dev:mock mode) is fully in control of the page before the loader
 * issues its first `/api/plugins` fetch. Doing this work as a
 * top-level module side effect would race the service-worker
 * activation: `requestIdleCallback` queues the discovery before SW
 * is ready, the first fetch escapes to the rsbuild proxy, and the
 * loader silently logs a warning instead of finding any plugins.
 *
 * Internal plugins (`core`, `metrics`, `molq`, `molvis`,
 * `tensorboard`) are statically imported here and registered
 * eagerly inside `bootPlugins()`. They do NOT appear in `/api/plugins`.
 * Third-party bundles discovered through Python's
 * `molexp.ui_plugins` entry-point group are the only consumers of
 * the dynamic-import loader path.
 *
 * Pure logic lives in `./loader.ts` so unit tests can exercise it
 * without dragging in `corePlugin`'s DOM-bound transitive imports.
 */

import { resetContributionRuntimeForTests } from "@/plugins/contribution-runtime";
import corePlugin from "@/plugins/core";
import {
  createLoaderState,
  type DynamicImport,
  discoverAndLoad,
  type LoaderState,
  loadRemotePlugin,
  type ManifestFetcher,
  registerPluginInstance,
  resetLoaderState,
  UI_PLUGIN_API_VERSION,
} from "@/plugins/loader";
import metricsPlugin from "@/plugins/metrics";
import molqPlugin from "@/plugins/molq";
import molvisPlugin from "@/plugins/molvis";
import tensorboardPlugin from "@/plugins/tensorboard";

/**
 * UI-plugin contract version frozen into this build. Defined in
 * `loader.ts` (where it's actually consumed); re-exported here as the
 * public API surface.
 */
export { UI_PLUGIN_API_VERSION };

const state: LoaderState = createLoaderState();
let booted = false;

/**
 * Eagerly install internal plugins, then schedule third-party
 * discovery against `/api/plugins`. Idempotent — calling twice is
 * a no-op.
 *
 * Must be called from the entry module **after** the application
 * has done any boot-time work that needs to land before plugin
 * discovery fires — most importantly, after `enableMocking()` in
 * dev:mock mode. Otherwise the loader's first fetch will race the
 * service-worker activation and silently fail.
 */
export const bootPlugins = (): void => {
  if (booted) {
    return;
  }
  booted = true;

  // Internal plugins are statically imported and registered eagerly —
  // they are part of the main bundle and do not appear in `/api/plugins`.
  registerPluginInstance(state, corePlugin);
  registerPluginInstance(state, metricsPlugin);
  registerPluginInstance(state, molqPlugin);
  registerPluginInstance(state, molvisPlugin);
  registerPluginInstance(state, tensorboardPlugin);

  if (typeof window === "undefined") {
    return;
  }

  const idle = (
    window as Window & {
      requestIdleCallback?: (cb: () => void) => void;
    }
  ).requestIdleCallback;
  if (idle) {
    idle(() => {
      void discoverAndLoad(state);
    });
  } else {
    setTimeout(() => {
      void discoverAndLoad(state);
    }, 0);
  }
};

export const ensureRemotePlugin = (
  id: string,
  manifestUrl: string,
  entryUrl: string,
): Promise<void> => {
  return loadRemotePlugin(state, { id, manifestUrl, entryUrl });
};

export const resetUiPluginsForTests = (): void => {
  resetLoaderState(state);
  resetContributionRuntimeForTests();
};

export const testHooks = {
  setDynamicImport: (impl: DynamicImport) => {
    state.dynamicImport = impl;
  },
  setFetchManifest: (impl: ManifestFetcher) => {
    state.fetchManifest = impl;
  },
  resetDynamicImport: () => {
    resetLoaderState(state);
  },
  discoverAndLoad: () => discoverAndLoad(state),
};
