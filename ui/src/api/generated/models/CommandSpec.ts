/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CommandParameterSpec } from './CommandParameterSpec';
/**
 * A single slash command — skill-backed or builtin.
 */
export type CommandSpec = {
    defaultPlanMode?: boolean;
    description?: string;
    isBuiltin?: boolean;
    name: string;
    parameters?: Array<CommandParameterSpec>;
    skillId?: (string | null);
    slashName: string;
};

