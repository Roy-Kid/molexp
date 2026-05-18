/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Public view of the workspace's LLM provider config — never the raw key.
 *
 * ``apiKeyPreview`` is a masked rendering ("sk-...1234"); ``apiKeySet``
 * is the boolean the UI uses to gate the "ready" indicator.
 */
export type AgentProviderResponse = {
    provider?: string;
    model?: string;
    baseUrl?: string;
    apiKeyPreview?: string;
    apiKeySet?: boolean;
    instructions?: string;
    supportedProviders?: Array<string>;
};
