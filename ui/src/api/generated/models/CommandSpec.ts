/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CommandParameterSpec } from './CommandParameterSpec';
/**
 * A single slash command — skill-backed or builtin.
 */
export type CommandSpec = {
    slashName: string;
    name: string;
    description?: string;
    parameters?: Array<CommandParameterSpec>;
    defaultPlanMode?: boolean;
    isBuiltin?: boolean;
    skillId?: (string | null);
};
