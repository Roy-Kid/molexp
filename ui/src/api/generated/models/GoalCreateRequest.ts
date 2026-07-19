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
    experimentId?: (string | null);
    /**
     * Replace the layered system prompt for this single session. Workspace and skill addenda are bypassed; the molexp built-in preamble is also dropped.
     */
    instructions_override?: (string | null);
    /**
     * Agent used for the first conversational turn. 'chat' runs the interactive agent; 'plan' runs the auditable nine-stage PlanMode pipeline.
     */
    mode?: GoalCreateRequest.mode;
    /**
     * Deprecated compatibility alias for mode='plan'. AgentTask routes run the auditable nine-stage Planning Agent when true.
     */
    plan_mode?: boolean;
    projectId?: (string | null);
    runId?: (string | null);
    /**
     * When the goal originates from a slash command, the underlying skill id (informational; the route still resolves the skill's instructions server-side).
     */
    skill_id?: (string | null);
    success_criteria?: Array<string>;
};
export namespace GoalCreateRequest {
    /**
     * Agent used for the first conversational turn. 'chat' runs the interactive agent; 'plan' runs the auditable nine-stage PlanMode pipeline.
     */
    export enum mode {
        CHAT = 'chat',
        PLAN = 'plan',
    }
}

