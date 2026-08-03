import { describe, expect, it } from "@rstest/core";

import type { ApiAgentProvider } from "@/app/state/api";

import { collectConfiguredModels, modelDisplayName } from "./ModelPicker";

const baseProvider = (overrides: Partial<ApiAgentProvider> = {}): ApiAgentProvider => ({
  provider: "openai",
  model: "openai:gpt-4o",
  baseUrl: "",
  apiKeyPreview: "",
  apiKeySet: true,
  instructions: "",
  supportedProviders: ["openai"],
  models: {
    cheap: "openai:gpt-4o-mini",
    default: "openai:gpt-4o",
    heavy: "openai:o1",
  },
  configurations: [],
  ...overrides,
});

describe("collectConfiguredModels", () => {
  it("uniques model + tier ids", () => {
    const ids = collectConfiguredModels(baseProvider());
    expect(ids.sort()).toEqual(["openai:gpt-4o", "openai:gpt-4o-mini", "openai:o1"].sort());
  });

  it("includes configuration tier models", () => {
    const ids = collectConfiguredModels(
      baseProvider({
        configurations: [
          {
            provider: "anthropic",
            models: {
              cheap: "anthropic:claude-haiku",
              default: "anthropic:claude-sonnet",
              heavy: "",
            },
            baseUrl: "",
            apiKeyPreview: "",
            apiKeySet: true,
          },
        ],
      }),
    );
    expect(ids).toContain("anthropic:claude-haiku");
    expect(ids).toContain("anthropic:claude-sonnet");
  });

  it("dedupes bare vs provider-qualified forms of the same model", () => {
    const ids = collectConfiguredModels(
      baseProvider({
        provider: "deepseek",
        model: "deepseek:deepseek-v4-flash",
        models: {
          cheap: "deepseek-v4-flash",
          default: "deepseek:deepseek-v4-flash",
          heavy: "deepseek-v4-pro",
        },
        configurations: [
          {
            provider: "deepseek",
            models: {
              cheap: "deepseek:deepseek-v4-flash",
              default: "deepseek-v4-pro",
              heavy: "deepseek:deepseek-v4-pro",
            },
            baseUrl: "",
            apiKeyPreview: "",
            apiKeySet: true,
          },
        ],
      }),
    );
    const displays = ids.map(modelDisplayName).map((s) => s.toLowerCase());
    expect(new Set(displays).size).toBe(displays.length);
    expect(displays).toContain("deepseek-v4-flash");
    expect(displays).toContain("deepseek-v4-pro");
    // Prefer qualified when both appear.
    expect(ids.some((id) => id === "deepseek:deepseek-v4-flash")).toBe(true);
  });
});

describe("modelDisplayName", () => {
  it("strips the provider prefix", () => {
    expect(modelDisplayName("openai:gpt-4o")).toBe("gpt-4o");
  });

  it("returns bare ids unchanged", () => {
    expect(modelDisplayName("gpt-4o")).toBe("gpt-4o");
  });
});
