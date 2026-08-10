/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * A saved skill (goal template + tool scope + system addendum).
 */
export type SkillResponse = {
    allowedTools?: Array<string>;
    builtin?: boolean;
    constraints?: Array<string>;
    createdAt?: string;
    defaultPlanMode?: boolean;
    deniedTools?: Array<string>;
    description?: string;
    goalTemplate: string;
    id: string;
    instructions?: string;
    name: string;
    requiresExitTool?: string;
    scope?: string;
    slashName?: string;
    successCriteria?: Array<string>;
    tags?: Array<string>;
    updatedAt?: string;
};

