/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Per-session system prompt breakdown for the inspector.
 */
export type AgentSystemPromptResponse = {
    base: string;
    workspaceInstructions?: string;
    skillInstructions?: string;
    sessionOverride?: (string | null);
    planMode?: boolean;
    effective: string;
};
