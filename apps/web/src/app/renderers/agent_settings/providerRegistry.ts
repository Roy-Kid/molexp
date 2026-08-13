/**
 * Provider registry — model-plugin-driven schema for the Provider tab.
 *
 * Per the agent-harness UI lockstep spec (§7.1 / §8) each model plugin
 * registers a `ProviderConfigValidator` describing its required and
 * optional fields. The UI reads these via `/api/agent/admin/providers`
 * and renders the form against them, so adding a new provider means
 * shipping a plugin — no UI edits.
 *
 * This module owns:
 *
 * - The registry payload type that the API returns.
 * - Pure helpers (`deriveProviderFields`, `findRegistryEntry`) that
 *   project the registry into a concrete form schema.
 *
 * Helpers live here (not in `state/api.ts`) so they can be unit-tested
 * from the node test environment without pulling the full api-client
 * singleton chain.
 */

export type ProviderFieldKind = "text" | "secret" | "url";

export interface ProviderFieldSpec {
  /** Form key, e.g. "api_key", "base_url", "model". */
  readonly key: string;
  /** Human-visible label. */
  readonly label: string;
  /** Input rendering hint. */
  readonly kind: ProviderFieldKind;
  /** Whether the field must be filled before save is allowed. */
  readonly required: boolean;
  /** Placeholder text rendered inside the empty input. */
  readonly placeholder?: string;
  /** Additional help shown under the field. */
  readonly help?: string;
}

export interface ProviderRegistryEntry {
  /** Stable identifier (e.g. `"openai"`, `"deepseek"`). */
  readonly name: string;
  /** Human-visible name shown in the provider picker. */
  readonly label: string;
  /** Help string shown next to the model field. */
  readonly modelHint: string;
  /** Field schema; the form is rendered straight from this list. */
  readonly fields: readonly ProviderFieldSpec[];
}

export interface ProviderRegistryResponse {
  readonly providers: readonly ProviderRegistryEntry[];
}

const EMPTY_FIELDS: readonly ProviderFieldSpec[] = [];

/**
 * Built-in registry used as the bootstrap fallback while the
 * `/api/agent/admin/providers` endpoint is loading or unavailable
 * (e.g. before the backend Phase 3 cutover lands the route in
 * production). Once the backend ships the route the UI overlays the
 * fetched response on top of these defaults.
 *
 * Keep these entries aligned with the validators that ship inside
 * `model_pydanticai`. New providers should land via plugins, not by
 * editing this table.
 */
export const DEFAULT_PROVIDER_REGISTRY: ProviderRegistryResponse = {
  providers: [
    {
      name: "anthropic",
      label: "Anthropic (Claude)",
      modelHint: "e.g. claude-sonnet-4-6, claude-opus-4-5",
      fields: [
        { key: "api_key", label: "API key", kind: "secret", required: true },
        { key: "model", label: "Model", kind: "text", required: true },
      ],
    },
    {
      name: "openai",
      label: "OpenAI",
      modelHint: "e.g. gpt-4o, gpt-4o-mini, o1-preview",
      fields: [
        { key: "api_key", label: "API key", kind: "secret", required: true },
        { key: "model", label: "Model", kind: "text", required: true },
        {
          key: "base_url",
          label: "Base URL (optional)",
          kind: "url",
          required: false,
          placeholder: "https://api.openai.com/v1 (leave blank for default)",
        },
      ],
    },
    {
      name: "google",
      label: "Google (Gemini)",
      modelHint: "e.g. gemini-2.0-flash, gemini-1.5-pro",
      fields: [
        { key: "api_key", label: "API key", kind: "secret", required: true },
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
        {
          key: "base_url",
          label: "Base URL (optional)",
          kind: "url",
          required: false,
          placeholder: "https://api.deepseek.com/v1 (leave blank for default)",
        },
      ],
    },
    {
      name: "openai-compatible",
      label: "OpenAI-compatible (proxy / Ollama / vLLM)",
      modelHint: "Any model exposed by the configured base_url endpoint",
      fields: [
        { key: "api_key", label: "API key", kind: "secret", required: true },
        { key: "model", label: "Model", kind: "text", required: true },
        {
          key: "base_url",
          label: "Base URL",
          kind: "url",
          required: true,
          placeholder: "http://localhost:11434/v1",
        },
      ],
    },
  ],
};

/**
 * Look up a registry entry by name. Returns `undefined` if the registry
 * has no validator for that provider — the caller decides whether that
 * is a hard error or a graceful fallback (typically: disable save and
 * show "Provider plugin not loaded").
 */
export const findRegistryEntry = (
  registry: ProviderRegistryResponse | null | undefined,
  providerName: string,
): ProviderRegistryEntry | undefined => {
  if (!registry) return undefined;
  return registry.providers.find((p) => p.name === providerName);
};

/**
 * Derive the form field schema for a given provider name, given a
 * registry response. Returns an empty array if the provider isn't in
 * the registry — callers can use that as a "schema not loaded" signal.
 */
export const deriveProviderFields = (
  registry: ProviderRegistryResponse | null | undefined,
  providerName: string,
): readonly ProviderFieldSpec[] => {
  const entry = findRegistryEntry(registry, providerName);
  return entry?.fields ?? EMPTY_FIELDS;
};

/** Returns true if the provider's validator declares a `base_url` field. */
export const supportsBaseUrl = (
  registry: ProviderRegistryResponse | null | undefined,
  providerName: string,
): boolean => {
  const fields = deriveProviderFields(registry, providerName);
  return fields.some((f) => f.key === "base_url");
};

/** Returns the placeholder for the `base_url` field, or empty string. */
export const baseUrlPlaceholder = (
  registry: ProviderRegistryResponse | null | undefined,
  providerName: string,
): string => {
  const fields = deriveProviderFields(registry, providerName);
  return fields.find((f) => f.key === "base_url")?.placeholder ?? "";
};

/**
 * Return the names of fields that are required and currently empty in
 * the form. The provider form uses this to disable the save button
 * with a precise tooltip ("missing: api_key, base_url").
 */
export const missingRequiredFields = (
  registry: ProviderRegistryResponse | null | undefined,
  providerName: string,
  values: Readonly<Record<string, string | undefined>>,
): readonly string[] => {
  const fields = deriveProviderFields(registry, providerName);
  return fields
    .filter((f) => f.required)
    .filter((f) => {
      const v = values[f.key];
      return v === undefined || v === "" || v === null;
    })
    .map((f) => f.key);
};
