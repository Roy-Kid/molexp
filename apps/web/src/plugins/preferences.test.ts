import { afterEach, beforeEach, describe, expect, it } from "@rstest/core";
import {
  getPluginEnabledMap,
  isPluginEnabled,
  replacePluginEnabledMap,
  resetPluginPreferencesForTests,
  setPluginEnabled,
} from "@/plugins/preferences";

describe("plugin preferences", () => {
  beforeEach(() => {
    resetPluginPreferencesForTests();
  });

  afterEach(() => {
    resetPluginPreferencesForTests();
  });

  it("defaults every plugin to enabled", () => {
    expect(isPluginEnabled("workflow")).toBe(true);
    expect(isPluginEnabled("molplot")).toBe(true);
    expect(getPluginEnabledMap()).toEqual({});
  });

  it("setPluginEnabled(false) disables a plugin", () => {
    setPluginEnabled("workflow", false);
    expect(isPluginEnabled("workflow")).toBe(false);
    expect(isPluginEnabled("molplot")).toBe(true);
    expect(getPluginEnabledMap()).toEqual({ workflow: false });
  });

  it("setPluginEnabled(true) re-enables a disabled plugin", () => {
    setPluginEnabled("molvis", false);
    setPluginEnabled("molvis", true);
    expect(isPluginEnabled("molvis")).toBe(true);
    expect(getPluginEnabledMap().molvis).toBe(true);
  });

  it("replacePluginEnabledMap overwrites the whole map", () => {
    replacePluginEnabledMap({ molq: false, deltaf: false });
    expect(isPluginEnabled("molq")).toBe(false);
    expect(isPluginEnabled("deltaf")).toBe(false);
    expect(isPluginEnabled("workflow")).toBe(true);
  });
});
