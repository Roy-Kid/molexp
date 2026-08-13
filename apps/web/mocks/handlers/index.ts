/**
 * Central export for all MSW mock handlers
 */

import { http, HttpResponse } from "msw";

import { agentHandlers } from "./agent";
import { agentAdminHandlers } from "./agent_admin";
import { assetHandlers } from "./assets";
import { authHandlers } from "./auth";
import { catalogHandlers } from "./catalog";
import { executionHandlers } from "./execution";
import { experimentHandlers } from "./experiments";
import { featureShowcaseHandlers } from "./feature_showcase";
import { molqHandlers } from "./molq";
import { projectHandlers } from "./projects";
import { registryHandlers } from "./registry";
import { runHandlers } from "./runs";
import { targetsHandlers } from "./targets";
import { workspaceTargetsHandlers } from "./workspace_targets";
import { workspaceHandlers } from "./workspace";

const unsupportedMockApiHandler = http.all(/\/api\/.*/, ({ request }) =>
    HttpResponse.json(
        {
            detail: `No mock handler is registered for ${request.method} ${new URL(request.url).pathname}`,
        },
        { status: 501 },
    ),
);

/**
 * All mock handlers combined
 */
export const handlers = [
    ...authHandlers,
    ...agentHandlers,
    ...agentAdminHandlers,
    ...assetHandlers,
    ...catalogHandlers,
    ...executionHandlers,
    ...experimentHandlers,
    ...featureShowcaseHandlers,
    ...molqHandlers,
    ...projectHandlers,
    ...registryHandlers,
    ...runHandlers,
    ...targetsHandlers,
    ...workspaceTargetsHandlers,
    ...workspaceHandlers,
    unsupportedMockApiHandler,
];
