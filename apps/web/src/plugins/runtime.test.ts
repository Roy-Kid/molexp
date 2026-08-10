/**
 * Tests for the plugin loader after the 07 split.
 *
 * The loader must:
 *
 *   - call `PluginsService.listPluginsApiPluginsGet()` to discover
 *     descriptors of shape `{id, manifestUrl, entryUrl}`;
 *   - for each descriptor, fetch its `manifestUrl` via
 *     `state.fetchManifest`, validate the body against
 *     `UiBundleManifest`, and check `manifest.api_version` matches
 *     `UI_PLUGIN_API_VERSION`;
 *   - dynamic-import the resolved entry URL and call the default
 *     export's `register()`;
 *   - isolate per-plugin failures (fetch, schema, version, import,
 *     malformed module) so other plugins keep loading.
 *
 * The built-in lazy compat table is gone — internal plugins
 * (`core`, `metrics`, `molplot`, `molq`, `molvis`, `tensorboard`) are
 * statically imported by `App.tsx`, not routed through `discoverAndLoad`.
 */

import { afterEach, beforeEach, describe, expect, it, rs } from "@rstest/core";

import { PluginsService } from "@/api/generated/services/PluginsService";
import {
  createLoaderState,
  discoverAndLoad,
  type LoaderState,
  UI_PLUGIN_API_VERSION,
} from "@/plugins/loader";
import type { PluginManifest, UiBundleManifest } from "@/plugins/types";

const makeDescriptor = (id: string): PluginManifest => ({
  id,
  manifestUrl: `/api/plugins/${id}/manifest.json`,
  entryUrl: `/api/plugins/${id}/index.js`,
});

const makeManifest = (id: string, overrides: Partial<UiBundleManifest> = {}): UiBundleManifest => ({
  id,
  name: id,
  version: "0.0.1",
  api_version: UI_PLUGIN_API_VERSION as "1",
  ...overrides,
});

describe("plugin loader (manifest-based)", () => {
  let state: LoaderState;
  let dynamicImportSpy: ReturnType<typeof rs.fn>;
  let fetchManifestSpy: ReturnType<typeof rs.fn>;
  let consoleWarnSpy: ReturnType<typeof rs.spyOn>;

  beforeEach(() => {
    state = createLoaderState();
    dynamicImportSpy = rs.fn();
    fetchManifestSpy = rs.fn();
    state.dynamicImport = dynamicImportSpy as never;
    state.fetchManifest = fetchManifestSpy as never;
    consoleWarnSpy = rs.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    rs.restoreAllMocks();
  });

  it("fetches manifest, checks api_version, dynamic-imports entry, calls register", async () => {
    const registerSpy = rs.fn();
    fetchManifestSpy.mockResolvedValueOnce(makeManifest("alpha"));
    dynamicImportSpy.mockResolvedValueOnce({
      default: { id: "alpha", register: registerSpy },
    });
    rs.spyOn(PluginsService, "listPluginsApiPluginsGet").mockResolvedValueOnce({
      plugins: [makeDescriptor("alpha")],
      total: 1,
    } as never);

    await discoverAndLoad(state);

    expect(fetchManifestSpy).toHaveBeenCalledTimes(1);
    expect(fetchManifestSpy).toHaveBeenCalledWith("/api/plugins/alpha/manifest.json");
    expect(dynamicImportSpy).toHaveBeenCalledTimes(1);
    expect(dynamicImportSpy).toHaveBeenCalledWith("/api/plugins/alpha/index.js");
    expect(registerSpy).toHaveBeenCalledTimes(1);
  });

  it("respects manifest.entry when it overrides the default index.js", async () => {
    const registerSpy = rs.fn();
    fetchManifestSpy.mockResolvedValueOnce(makeManifest("alpha", { entry: "main.js" }));
    dynamicImportSpy.mockResolvedValueOnce({
      default: { id: "alpha", register: registerSpy },
    });
    rs.spyOn(PluginsService, "listPluginsApiPluginsGet").mockResolvedValueOnce({
      plugins: [makeDescriptor("alpha")],
      total: 1,
    } as never);

    await discoverAndLoad(state);

    expect(dynamicImportSpy).toHaveBeenCalledTimes(1);
    const calledWith = dynamicImportSpy.mock.calls[0]?.[0] as string;
    // It must use main.js, not index.js — exact resolution algorithm
    // is the loader's choice, but the URL must end in main.js.
    expect(calledWith.endsWith("main.js")).toBe(true);
    expect(registerSpy).toHaveBeenCalledTimes(1);
  });

  it("skips with warning when manifest.api_version mismatches", async () => {
    fetchManifestSpy.mockResolvedValueOnce({
      ...makeManifest("alpha"),
      api_version: "999" as never,
    });
    rs.spyOn(PluginsService, "listPluginsApiPluginsGet").mockResolvedValueOnce({
      plugins: [makeDescriptor("alpha")],
      total: 1,
    } as never);

    await discoverAndLoad(state);

    expect(dynamicImportSpy).not.toHaveBeenCalled();
    expect(consoleWarnSpy).toHaveBeenCalled();
    expect(state.installed.has("alpha")).toBe(false);
  });

  it("survives a failing manifest fetch and lets other plugins continue", async () => {
    const registerSpy = rs.fn();
    fetchManifestSpy.mockImplementation(async (url: string) => {
      if (url.includes("broken")) {
        throw new Error("network gone");
      }
      return makeManifest("good");
    });
    dynamicImportSpy.mockResolvedValueOnce({
      default: { id: "good", register: registerSpy },
    });
    rs.spyOn(PluginsService, "listPluginsApiPluginsGet").mockResolvedValueOnce({
      plugins: [makeDescriptor("broken"), makeDescriptor("good")],
      total: 2,
    } as never);

    await expect(discoverAndLoad(state)).resolves.not.toThrow();
    expect(dynamicImportSpy).toHaveBeenCalledTimes(1);
    expect(dynamicImportSpy).toHaveBeenCalledWith("/api/plugins/good/index.js");
    expect(registerSpy).toHaveBeenCalledTimes(1);
    expect(consoleWarnSpy).toHaveBeenCalled();
  });

  it("survives a failing dynamic import", async () => {
    fetchManifestSpy.mockResolvedValueOnce(makeManifest("broken"));
    dynamicImportSpy.mockRejectedValueOnce(new Error("module gone"));
    rs.spyOn(PluginsService, "listPluginsApiPluginsGet").mockResolvedValueOnce({
      plugins: [makeDescriptor("broken")],
      total: 1,
    } as never);

    await expect(discoverAndLoad(state)).resolves.not.toThrow();
    expect(consoleWarnSpy).toHaveBeenCalled();
    expect(state.installed.has("broken")).toBe(false);
  });

  it("survives a failing /api/plugins fetch", async () => {
    rs.spyOn(PluginsService, "listPluginsApiPluginsGet").mockRejectedValueOnce(
      new Error("server down"),
    );

    await expect(discoverAndLoad(state)).resolves.not.toThrow();
    expect(fetchManifestSpy).not.toHaveBeenCalled();
    expect(dynamicImportSpy).not.toHaveBeenCalled();
  });

  it("ignores remote modules whose default export is malformed", async () => {
    fetchManifestSpy.mockResolvedValueOnce(makeManifest("malformed"));
    dynamicImportSpy.mockResolvedValueOnce({ default: { foo: "bar" } });
    rs.spyOn(PluginsService, "listPluginsApiPluginsGet").mockResolvedValueOnce({
      plugins: [makeDescriptor("malformed")],
      total: 1,
    } as never);

    await discoverAndLoad(state);

    expect(state.installed.has("malformed")).toBe(false);
    expect(consoleWarnSpy).toHaveBeenCalled();
  });

  it("rejects manifest body that fails the UiBundleManifest schema", async () => {
    // Missing `version` field — invalid manifest.
    fetchManifestSpy.mockResolvedValueOnce({
      id: "broken",
      name: "Broken",
      api_version: "1",
    } as never);
    rs.spyOn(PluginsService, "listPluginsApiPluginsGet").mockResolvedValueOnce({
      plugins: [makeDescriptor("broken")],
      total: 1,
    } as never);

    await discoverAndLoad(state);

    expect(dynamicImportSpy).not.toHaveBeenCalled();
    expect(consoleWarnSpy).toHaveBeenCalled();
    expect(state.installed.has("broken")).toBe(false);
  });
});
