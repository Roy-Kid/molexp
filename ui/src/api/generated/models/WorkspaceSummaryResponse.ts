/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { HealthFlagResponse } from './HealthFlagResponse';
import type { KnowledgeRefResponse } from './KnowledgeRefResponse';
import type { NextActionResponse } from './NextActionResponse';
import type { RunRefResponse } from './RunRefResponse';
import type { WorkspaceRefResponse } from './WorkspaceRefResponse';
/**
 * Camel-cased HTTP view of the read-only ``WorkspaceSummary``.
 */
export type WorkspaceSummaryResponse = {
    counts: Record<string, number>;
    failedRuns?: Array<RunRefResponse>;
    headline: string;
    healthFlags?: Array<HealthFlagResponse>;
    nextActions?: Array<NextActionResponse>;
    openQuestions?: Array<KnowledgeRefResponse>;
    relevantKnowledge?: Array<KnowledgeRefResponse>;
    runningRuns?: Array<RunRefResponse>;
    workspace: WorkspaceRefResponse;
};

