/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CustomToolHttpInvokerRequest } from './CustomToolHttpInvokerRequest';
/**
 * Create a user/workspace-tier tool declaration.
 */
export type CustomToolCreateRequest = {
    /**
     * Tool name as the LLM will see it
     */
    name: string;
    /**
     * Tool description shown to the LLM during selection
     */
    description?: string;
    /**
     * JSON Schema describing the tool arguments
     */
    parametersSchema?: Record<string, any>;
    requiresApproval?: boolean;
    category?: CustomToolCreateRequest.category;
    mutates?: boolean;
    /**
     * Invocation spec. Initial UI release supports only 'http'; package-shipped tools use the 'python' kind via @default_tool.
     */
    invoker: CustomToolHttpInvokerRequest;
    scope?: CustomToolCreateRequest.scope;
    /**
     * Optional explicit id; defaults to the tool's name.
     */
    tool_id?: (string | null);
};
export namespace CustomToolCreateRequest {
    export enum category {
        WORKSPACE = 'workspace',
        WORKFLOW = 'workflow',
        CHAT = 'chat',
        CONTROL = 'control',
    }
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

