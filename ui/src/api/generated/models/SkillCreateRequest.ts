/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type SkillCreateRequest = {
    /**
     * Display name
     */
    name: string;
    /**
     * Goal description, may contain {{param}} placeholders
     */
    goal_template: string;
    /**
     * Long description
     */
    description?: string;
    /**
     * Optional slash command id (e.g. 'plot-energy'). When set, the skill is invokable from the chat input as /<slash_name>. Reserved names: plan, clear, model, help.
     */
    slash_name?: string;
    /**
     * System prompt addendum applied when this skill launches a session
     */
    instructions?: string;
    /**
     * Sessions launched from this skill default to plan mode
     */
    default_plan_mode?: boolean;
    constraints?: Array<string>;
    success_criteria?: Array<string>;
    tags?: Array<string>;
};

