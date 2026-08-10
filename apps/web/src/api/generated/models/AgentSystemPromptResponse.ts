/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Per-session system prompt breakdown for the inspector.
 */
export type AgentSystemPromptResponse = {
    base: string;
    effective: string;
    planMode?: boolean;
    sessionOverride?: (string | null);
    skillInstructions?: string;
    workspaceInstructions?: string;
};

