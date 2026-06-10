/**
 * Behavioral tests for the PUT workflow-document mock — the
 * CreateWorkflowDialog empty-IR seed path (ui-creation-entries).
 *
 * Resolves requests against the handler list via msw's getResponse();
 * the mock db is module-level state, so each test seeds its own experiment.
 */

import { describe, expect, it } from "@rstest/core";
import { getResponse } from "msw";

import type { ApiExperimentResponse } from "../../../src/app/types";
import { getExperiment, setExperiment } from "../../db";
import { experimentHandlers } from "../experiments";

// msw resolves the handlers' relative paths against location, which the
// node test env lacks — pin it before any matching happens.
Reflect.set(globalThis, "location", new URL("http://localhost/"));

const EMPTY_IR = { task_configs: [], links: [] };

const seedExperiment = (id: string, projectId: string): ApiExperimentResponse => {
  const experiment: ApiExperimentResponse = {
    id,
    projectId,
    name: id,
    description: "",
    workflow: null,
    workflowType: "yaml",
    gitCommit: null,
    parameterSpace: {},
    runCount: 0,
    runs: [],
    created: new Date().toISOString(),
  };
  setExperiment(experiment);
  return experiment;
};

const putRequest = (projectId: string, experimentId: string): Request =>
  new Request(
    `http://localhost/api/projects/${projectId}/experiments/${experimentId}/workflow`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document: EMPTY_IR }),
    },
  );

describe("PUT /api/projects/:projectId/experiments/:experimentId/workflow mock", () => {
  it("returns the WorkflowDocumentResponse shape for the empty-IR seed", async () => {
    seedExperiment("wf-seed-exp", "proj-wf");
    const response = await getResponse(experimentHandlers, putRequest("proj-wf", "wf-seed-exp"));
    expect(response?.status).toBe(200);
    const body = await response?.json();
    expect(body).toEqual({
      project_id: "proj-wf",
      experiment_id: "wf-seed-exp",
      document: EMPTY_IR,
    });
  });

  it("persists the document onto the experiment workflow field", async () => {
    seedExperiment("wf-persist-exp", "proj-wf");
    await getResponse(experimentHandlers, putRequest("proj-wf", "wf-persist-exp"));
    const stored = getExperiment("wf-persist-exp");
    const parsed = JSON.parse(stored?.workflow ?? "null") as {
      task_configs: unknown[];
      links: unknown[];
    };
    expect(parsed.task_configs).toEqual([]);
    expect(parsed.links).toEqual([]);
  });

  it("404s an unknown experiment", async () => {
    const response = await getResponse(experimentHandlers, putRequest("proj-wf", "wf-ghost"));
    expect(response?.status).toBe(404);
  });
});
