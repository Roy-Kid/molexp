/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ArtifactRefResponse } from './ArtifactRefResponse';
import type { ContextFocusResponse } from './ContextFocusResponse';
import type { ExperimentRefResponse } from './ExperimentRefResponse';
import type { HealthFlagResponse } from './HealthFlagResponse';
import type { KnowledgeRefResponse } from './KnowledgeRefResponse';
import type { ProjectRefResponse } from './ProjectRefResponse';
import type { RunRefResponse } from './RunRefResponse';
import type { WorkflowRefResponse } from './WorkflowRefResponse';
import type { WorkspaceRefResponse } from './WorkspaceRefResponse';
/**
 * Camel-cased HTTP view of the canonical ``WorkspaceContext`` read-model.
 */
export type WorkspaceContextResponse = {
    artifacts?: Array<ArtifactRefResponse>;
    experiments?: Array<ExperimentRefResponse>;
    failedRuns?: Array<RunRefResponse>;
    focus: ContextFocusResponse;
    knowledge?: Array<KnowledgeRefResponse>;
    openQuestions?: Array<KnowledgeRefResponse>;
    projects?: Array<ProjectRefResponse>;
    recentRuns?: Array<RunRefResponse>;
    runningRuns?: Array<RunRefResponse>;
    staleOrMissing?: Array<HealthFlagResponse>;
    workflows?: Array<WorkflowRefResponse>;
    workspace: WorkspaceRefResponse;
};

