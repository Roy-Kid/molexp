/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type SkillUpdateRequest = {
    name?: (string | null);
    goal_template?: (string | null);
    description?: (string | null);
    slash_name?: (string | null);
    instructions?: (string | null);
    default_plan_mode?: (boolean | null);
    constraints?: (Array<string> | null);
    success_criteria?: (Array<string> | null);
    tags?: (Array<string> | null);
    allowed_tools?: (Array<string> | null);
    denied_tools?: (Array<string> | null);
    requires_exit_tool?: (string | null);
};
