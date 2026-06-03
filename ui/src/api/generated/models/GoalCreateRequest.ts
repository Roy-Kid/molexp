/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type GoalCreateRequest = {
    constraints?: Record<string, any>;
    /**
     * Natural language goal description
     */
    description: string;
    /**
     * Replace the layered system prompt for this single session. Workspace and skill addenda are bypassed; the molexp built-in preamble is also dropped.
     */
    instructions_override?: (string | null);
    /**
     * When true, the runtime registers only read-only tools and asks the agent to emit a structured plan instead of executing.
     */
    plan_mode?: boolean;
    /**
     * When the goal originates from a slash command, the underlying skill id (informational; the route still resolves the skill's instructions server-side).
     */
    skill_id?: (string | null);
    success_criteria?: Array<string>;
};

