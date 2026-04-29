/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Materialize a skill into a Goal and start a session.
 */
export type SkillLaunchRequest = {
    parameters?: Record<string, any>;
    /**
     * Override the skill's ``default_plan_mode``. ``None`` (default) honors the skill's setting.
     */
    plan_mode?: (boolean | null);
};

