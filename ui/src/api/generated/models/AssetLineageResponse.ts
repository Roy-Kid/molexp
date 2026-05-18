/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AssetLineageNode } from './AssetLineageNode';
/**
 * Upstream + downstream neighbours of an asset in the lineage DAG.
 *
 * ``ancestors`` is the transitive set of upstream asset_ids reached
 * by walking ``producer.inputs`` in reverse; ``descendants`` is the
 * transitive forward set. The starting asset is excluded from both.
 */
export type AssetLineageResponse = {
    asset_id: string;
    ancestors?: Array<AssetLineageNode>;
    descendants?: Array<AssetLineageNode>;
};
