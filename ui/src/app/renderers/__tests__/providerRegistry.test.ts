import { describe, expect, it } from "@rstest/core";

import {
  deriveProviderFields,
  findRegistryEntry,
  missingRequiredFields,
  type ProviderRegistryResponse,
} from "../agent_settings/providerRegistry";

// ac-005 — provider form is registry-driven, not hard-coded.
//
// The fixture mocks two providers with deliberately different validator
// schemas: openai-compatible requires `base_url`, deepseek does not.
// The derivation must yield different field sets for each provider.

const fixture: ProviderRegistryResponse = {
  providers: [
    {
      name: "openai-compatible",
      label: "OpenAI-compatible",
      modelHint: "Any model exposed by the configured base_url endpoint",
      fields: [
        { key: "api_key", label: "API key", kind: "secret", required: true },
        {
          key: "base_url",
          label: "Base URL",
          kind: "url",
          required: true,
          placeholder: "http://localhost:11434/v1",
        },
        { key: "model", label: "Model", kind: "text", required: true },
      ],
    },
    {
      name: "deepseek",
      label: "DeepSeek",
      modelHint: "e.g. deepseek-chat, deepseek-reasoner",
      fields: [
        { key: "api_key", label: "API key", kind: "secret", required: true },
        { key: "model", label: "Model", kind: "text", required: true },
      ],
    },
  ],
};

describe("findRegistryEntry", () => {
  it("returns the entry whose name matches", () => {
    const entry = findRegistryEntry(fixture, "deepseek");
    expect(entry?.label).toBe("DeepSeek");
  });

  it("returns undefined for unknown provider names", () => {
    expect(findRegistryEntry(fixture, "anthropic")).toBeUndefined();
  });

  it("treats null / undefined registries as empty", () => {
    expect(findRegistryEntry(null, "openai-compatible")).toBeUndefined();
    expect(findRegistryEntry(undefined, "openai-compatible")).toBeUndefined();
  });
});

describe("deriveProviderFields", () => {
  it("returns the validator's full field list for a known provider", () => {
    const fields = deriveProviderFields(fixture, "openai-compatible");
    expect(fields.map((f) => f.key)).toEqual(["api_key", "base_url", "model"]);
  });

  it("renders distinct field sets for providers with different validators", () => {
    const compat = deriveProviderFields(fixture, "openai-compatible").map((f) => f.key);
    const deepseek = deriveProviderFields(fixture, "deepseek").map((f) => f.key);
    expect(compat).toContain("base_url");
    expect(deepseek).not.toContain("base_url");
    expect(compat).not.toEqual(deepseek);
  });

  it("returns an empty list when the provider is not in the registry", () => {
    expect(deriveProviderFields(fixture, "google")).toEqual([]);
  });
});

describe("missingRequiredFields", () => {
  it("returns the keys of required-but-empty fields", () => {
    const missing = missingRequiredFields(fixture, "openai-compatible", {
      api_key: "secret",
      base_url: "",
      model: undefined,
    });
    expect(missing).toEqual(["base_url", "model"]);
  });

  it("returns an empty list when all required fields are filled", () => {
    const missing = missingRequiredFields(fixture, "deepseek", {
      api_key: "secret",
      model: "deepseek-chat",
    });
    expect(missing).toEqual([]);
  });

  it("does not report optional fields as missing", () => {
    const registryWithOptional: ProviderRegistryResponse = {
      providers: [
        {
          name: "p",
          label: "P",
          modelHint: "",
          fields: [
            { key: "api_key", label: "API key", kind: "secret", required: true },
            { key: "org", label: "Organization", kind: "text", required: false },
          ],
        },
      ],
    };
    const missing = missingRequiredFields(registryWithOptional, "p", { api_key: "k" });
    expect(missing).toEqual([]);
  });
});
