/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ProviderConfigurationResponse } from './ProviderConfigurationResponse';
/**
 * The Settings page's provider view — never carries a key value.
 */
export type ProviderResponse = {
    apiKeyPreview: string;
    apiKeySet: boolean;
    baseUrl: string;
    configurations?: Array<ProviderConfigurationResponse>;
    instructions: string;
    model: string;
    provider: string;
    supportedProviders?: Array<string>;
};

