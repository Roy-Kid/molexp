/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TierModelsResponse } from './TierModelsResponse';
/**
 * One provider's independently persisted credentials and tier models.
 */
export type ProviderConfigurationResponse = {
    apiKeyPreview?: string;
    apiKeySet?: boolean;
    baseUrl?: string;
    models?: TierModelsResponse;
    provider: string;
};

