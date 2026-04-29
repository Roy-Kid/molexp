/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Patch the workspace's LLM provider config.
 *
 * Any field left as ``None`` is preserved. Pass ``api_key=""`` to clear
 * the stored key (e.g. when switching to env-var-only auth).
 */
export type AgentProviderUpdateRequest = {
    /**
     * One of: 'anthropic', 'openai', 'google', 'openai-compatible'
     */
    provider?: (string | null);
    /**
     * Provider-specific model name
     */
    model?: (string | null);
    /**
     * Set to clear by passing empty string
     */
    api_key?: (string | null);
    /**
     * Optional override for proxy/self-hosted endpoints
     */
    base_url?: (string | null);
    /**
     * Workspace-default system prompt addendum. Pass an empty string to clear; ``None`` leaves the existing value untouched.
     */
    instructions?: (string | null);
};

