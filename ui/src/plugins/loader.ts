/**
 * Pure plugin-loader logic for third-party UI bundles.
 *
 * Flow:
 *   1. Fetch ``GET /api/plugins`` — returns a list of `PluginManifest`
 *      descriptors, each carrying the bundle id and two URLs.
 *   2. For each descriptor, fetch its `manifestUrl` and validate the
 *      body against the {@link UiBundleManifest} schema.
 *   3. Skip the bundle (with a warning) when the manifest's
 *      `api_version` does not match the build-time
 *      `UI_PLUGIN_API_VERSION` constant.
 *   4. Resolve the entry URL — `manifest.entry` (default `index.js`)
 *      against the descriptor's `entryUrl` directory — and dynamic-
 *      import it through `state.dynamicImport`.
 *   5. Validate the module's default export shape (`{id, register}`)
 *      and call `register()`.
 *
 * Failures at every step are isolated with `console.warn` so a single
 * broken third-party bundle cannot block other plugins.
 *
 * Kept separate from `runtime.ts` so unit tests can drive the flow
 * without pulling in `corePlugin`'s DOM-bound transitive imports.
 */

import { PluginsService } from "@/api/generated/services/PluginsService";
import type { PluginManifest, UiBundleManifest, UiPluginModule } from "@/plugins/types";

/**
 * UI-plugin contract version frozen into this build. Each third-party
 * `manifest.json` declares its own `api_version`; the loader skips
 * bundles whose value does not match this constant. Re-exported from
 * `runtime.ts` for backwards-compat callers.
 */
export const UI_PLUGIN_API_VERSION = "1";

export type DynamicImport = (specifier: string) => Promise<unknown>;

export type ManifestFetcher = (url: string) => Promise<UiBundleManifest>;

const defaultDynamicImport: DynamicImport = (specifier) =>
  Function("s", "return import(s)")(specifier);

const defaultFetchManifest: ManifestFetcher = async (url) => {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`fetch ${url} -> HTTP ${response.status}`);
  }
  return (await response.json()) as UiBundleManifest;
};

export interface LoaderState {
  installed: Set<string>;
  remotePromises: Map<string, Promise<void>>;
  dynamicImport: DynamicImport;
  fetchManifest: ManifestFetcher;
}

export const createLoaderState = (): LoaderState => ({
  installed: new Set<string>(),
  remotePromises: new Map<string, Promise<void>>(),
  dynamicImport: defaultDynamicImport,
  fetchManifest: defaultFetchManifest,
});

export const registerPluginInstance = (state: LoaderState, plugin: UiPluginModule): void => {
  if (state.installed.has(plugin.id)) {
    return;
  }
  state.installed.add(plugin.id);
  try {
    const result = plugin.register();
    if (result instanceof Promise) {
      result.catch((error) => {
        console.warn(`[plugins] "${plugin.id}" register() rejected:`, error);
      });
    }
  } catch (error) {
    console.warn(`[plugins] "${plugin.id}" register() threw:`, error);
  }
};

const looksLikeUiBundleManifest = (mod: unknown): mod is UiBundleManifest => {
  if (!mod || typeof mod !== "object") {
    return false;
  }
  const m = mod as Record<string, unknown>;
  return (
    typeof m.id === "string" &&
    typeof m.name === "string" &&
    typeof m.version === "string" &&
    typeof m.api_version === "string"
  );
};

const looksLikePluginModule = (mod: unknown): mod is { default: UiPluginModule } => {
  if (!mod || typeof mod !== "object") {
    return false;
  }
  const candidate = (mod as { default?: unknown }).default;
  return (
    typeof candidate === "object" &&
    candidate !== null &&
    typeof (candidate as { id?: unknown }).id === "string" &&
    typeof (candidate as { register?: unknown }).register === "function"
  );
};

const resolveEntryUrl = (descriptor: PluginManifest, manifest: UiBundleManifest): string => {
  const entry = manifest.entry;
  if (!entry || entry === "index.js") {
    return descriptor.entryUrl;
  }
  // Resolve `entry` relative to the bundle's directory — that is, the
  // directory containing `descriptor.entryUrl`. We don't use the URL
  // constructor because relative resolution against a relative URL
  // requires a base; a string slice is unambiguous and bundler-safe.
  const lastSlash = descriptor.entryUrl.lastIndexOf("/");
  const dir = lastSlash >= 0 ? descriptor.entryUrl.slice(0, lastSlash + 1) : "";
  return `${dir}${entry}`;
};

export const loadRemotePlugin = (state: LoaderState, descriptor: PluginManifest): Promise<void> => {
  const cached = state.remotePromises.get(descriptor.id);
  if (cached) {
    return cached;
  }
  const promise = (async () => {
    let manifest: UiBundleManifest;
    try {
      const body = await state.fetchManifest(descriptor.manifestUrl);
      if (!looksLikeUiBundleManifest(body)) {
        console.warn(
          `[plugins] manifest at ${descriptor.manifestUrl} did not match UiBundleManifest schema; skipping ${descriptor.id}`,
        );
        return;
      }
      manifest = body;
    } catch (error) {
      console.warn(`[plugins] failed to fetch manifest for "${descriptor.id}":`, error);
      state.remotePromises.delete(descriptor.id);
      return;
    }

    if (manifest.api_version !== UI_PLUGIN_API_VERSION) {
      console.warn(
        `[plugins] "${descriptor.id}" targets api_version=${manifest.api_version} but UI build expects ${UI_PLUGIN_API_VERSION}; skipping`,
      );
      return;
    }

    const entryUrl = resolveEntryUrl(descriptor, manifest);
    try {
      const mod = await state.dynamicImport(entryUrl);
      if (!looksLikePluginModule(mod)) {
        console.warn(
          `[plugins] remote plugin "${descriptor.id}" loaded from ${entryUrl} did not export a UiPluginModule`,
        );
        return;
      }
      registerPluginInstance(state, mod.default);
    } catch (error) {
      console.warn(
        `[plugins] failed to load remote plugin "${descriptor.id}" from ${entryUrl}:`,
        error,
      );
      state.remotePromises.delete(descriptor.id);
    }
  })();
  state.remotePromises.set(descriptor.id, promise);
  return promise;
};

/**
 * Fetch `/api/plugins` and dispatch each descriptor through
 * {@link loadRemotePlugin}. Failures are isolated.
 */
export const discoverAndLoad = async (state: LoaderState): Promise<void> => {
  let listing: Awaited<ReturnType<typeof PluginsService.listPluginsApiPluginsGet>>;
  try {
    listing = await PluginsService.listPluginsApiPluginsGet();
  } catch (error) {
    console.warn("[plugins] /api/plugins fetch failed:", error);
    return;
  }
  await Promise.all(listing.plugins.map((descriptor) => loadRemotePlugin(state, descriptor)));
};

export const resetLoaderState = (state: LoaderState): void => {
  state.installed.clear();
  state.remotePromises.clear();
  state.dynamicImport = defaultDynamicImport;
  state.fetchManifest = defaultFetchManifest;
};
