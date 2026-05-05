/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CustomToolHttpInvokerRequest } from './CustomToolHttpInvokerRequest';
/**
 * Patch a user/workspace-tier tool declaration. Omitted fields stay.
 */
export type CustomToolUpdateRequest = {
    name?: (string | null);
    description?: (string | null);
    parametersSchema?: (Record<string, any> | null);
    requiresApproval?: (boolean | null);
    category?: ('workspace' | 'workflow' | 'chat' | 'control' | 'web' | null);
    mutates?: (boolean | null);
    invoker?: (CustomToolHttpInvokerRequest | null);
};

