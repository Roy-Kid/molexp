/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * A saved skill (goal template + tool scope + system addendum).
 */
export type SkillResponse = {
    id: string;
    name: string;
    description?: string;
    goalTemplate: string;
    slashName?: string;
    instructions?: string;
    defaultPlanMode?: boolean;
    constraints?: Array<string>;
    successCriteria?: Array<string>;
    tags?: Array<string>;
    allowedTools?: Array<string>;
    deniedTools?: Array<string>;
    requiresExitTool?: string;
    builtin?: boolean;
    scope?: string;
    createdAt?: string;
    updatedAt?: string;
};

