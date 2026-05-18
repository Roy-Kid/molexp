/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CustomToolHttpInvokerResponse } from './CustomToolHttpInvokerResponse';
import type { CustomToolPythonInvokerResponse } from './CustomToolPythonInvokerResponse';
/**
 * Single user/workspace/registration-tier tool record.
 *
 * Mirrors the `AgentToolResponse` shape but adds the persistence
 * metadata (`scope`, `shadowed`, `valid`, `createdAt`, `updatedAt`)
 * needed for tier-aware listing and inline error reporting.
 */
export type CustomToolResponse = {
    id: string;
    name: string;
    description?: string;
    category?: CustomToolResponse.category;
    mutates?: boolean;
    requiresApproval?: boolean;
    parametersSchema?: Record<string, any>;
    invoker: (CustomToolHttpInvokerResponse | CustomToolPythonInvokerResponse);
    scope?: CustomToolResponse.scope;
    shadowed?: boolean;
    valid?: boolean;
    invalidReason?: string;
    builtin?: boolean;
    createdAt?: string;
    updatedAt?: string;
};
export namespace CustomToolResponse {
    export enum category {
        WORKSPACE = 'workspace',
        WORKFLOW = 'workflow',
        CHAT = 'chat',
        CONTROL = 'control',
        WEB = 'web',
    }
    export enum scope {
        NATIVE = 'native',
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}
