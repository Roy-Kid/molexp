/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { EntityBacklinkRow } from './EntityBacklinkRow';
/**
 * ``GET /knowledge/entity-backlinks`` — who cites this entity?
 */
export type EntityBacklinksResponse = {
    backlinks: Array<EntityBacklinkRow>;
    entity: string;
};

