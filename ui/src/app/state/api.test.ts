/**
 * Tests for app/state/api.ts — pure mapping functions.
 *
 * Per project convention:
 * - describe('functionName') wraps each exported function
 * - it('...') covers one behaviour per case
 * - shared mock data lives in src/__fixtures__/api.ts
 */

import { describe, expect, it } from "@rstest/core";
import {
  fixtureAsset,
  fixtureExperiment,
  fixtureExperimentNoDescription,
  fixtureProject,
  fixtureProjectNoDescription,
  fixtureRun,
  fixtureRunCancelled,
  fixtureRunFailed,
  fixtureRunPending,
} from "@/__fixtures__/api";
import {
  buildEmptySnapshot,
  emptyConsoleEntries,
  mapAgentSessions,
  mapAssets,
  mapExperiments,
  mapProjects,
  mapRuns,
  mapWorkflows,
  mapWorkspaceTree,
} from "@/app/state/api";

describe("buildEmptySnapshot", () => {
  it("returns empty arrays for all collections", () => {
    const snap = buildEmptySnapshot();
    expect(snap.projects).toEqual([]);
    expect(snap.experiments).toEqual([]);
    expect(snap.runs).toEqual([]);
    expect(snap.assets).toEqual([]);
    expect(snap.workflows).toEqual([]);
    expect(snap.agentSessions).toEqual([]);
    expect(snap.consoleEntries).toEqual([]);
  });

  it("sets workspaceRoot to null", () => {
    expect(buildEmptySnapshot().workspaceRoot).toBeNull();
  });
});

describe("emptyConsoleEntries", () => {
  it("returns an empty array", () => {
    expect(emptyConsoleEntries()).toEqual([]);
  });
});

describe("mapProjects", () => {
  it("maps id and name from API response", () => {
    const [result] = mapProjects([fixtureProject]);
    expect(result.id).toBe("proj-alpha");
    expect(result.name).toBe("Alpha Project");
  });

  it("uses description as summary when present", () => {
    const [result] = mapProjects([fixtureProject]);
    expect(result.summary).toBe("First project");
  });

  it("falls back to 'No description' when description is absent", () => {
    const [result] = mapProjects([fixtureProjectNoDescription]);
    expect(result.summary).toBe("No description");
  });

  it("sets status to 'active'", () => {
    const [result] = mapProjects([fixtureProject]);
    expect(result.status).toBe("active");
  });

  it("maps updatedAt from created timestamp", () => {
    const [result] = mapProjects([fixtureProject]);
    expect(result.updatedAt).toBe(fixtureProject.created);
  });

  it("maps an empty array to an empty array", () => {
    expect(mapProjects([])).toEqual([]);
  });

  it("preserves order of multiple projects", () => {
    const results = mapProjects([fixtureProject, fixtureProjectNoDescription]);
    expect(results[0].id).toBe("proj-alpha");
    expect(results[1].id).toBe("proj-beta");
  });
});

describe("mapExperiments", () => {
  it("maps id, name, and projectId", () => {
    const [result] = mapExperiments("proj-alpha", [fixtureExperiment]);
    expect(result.id).toBe("exp-001");
    expect(result.name).toBe("Baseline");
    expect(result.projectId).toBe("proj-alpha");
  });

  it("uses description as summary when present", () => {
    const [result] = mapExperiments("proj-alpha", [fixtureExperiment]);
    expect(result.summary).toBe("Baseline experiment");
  });

  it("falls back to workflow path when description is absent", () => {
    const [result] = mapExperiments("proj-alpha", [fixtureExperimentNoDescription]);
    expect(result.summary).toBe("variant.py");
  });

  it("sets workflowFile from workflow field", () => {
    const [result] = mapExperiments("proj-alpha", [fixtureExperiment]);
    expect(result.workflowFile).toBe("workflow.py");
  });

  it("maps an empty array to an empty array", () => {
    expect(mapExperiments("proj-alpha", [])).toEqual([]);
  });
});

describe("mapRuns", () => {
  it("maps id and name from runId", () => {
    const [result] = mapRuns("proj-alpha", "exp-001", [fixtureRun]);
    expect(result.id).toBe("run-abc");
    expect(result.name).toBe("run-abc");
  });

  it("maps projectId and experimentId", () => {
    const [result] = mapRuns("proj-alpha", "exp-001", [fixtureRun]);
    expect(result.projectId).toBe("proj-alpha");
    expect(result.experimentId).toBe("exp-001");
  });

  it("maps status 'succeeded'", () => {
    const [result] = mapRuns("p", "e", [fixtureRun]);
    expect(result.status).toBe("succeeded");
  });

  it("maps status 'pending'", () => {
    const [result] = mapRuns("p", "e", [fixtureRunPending]);
    expect(result.status).toBe("pending");
  });

  it("maps status 'failed'", () => {
    const [result] = mapRuns("p", "e", [fixtureRunFailed]);
    expect(result.status).toBe("failed");
  });

  it("maps status 'cancelled'", () => {
    const [result] = mapRuns("p", "e", [fixtureRunCancelled]);
    expect(result.status).toBe("cancelled");
  });

  it("uses finished timestamp for updatedAt when available", () => {
    const [result] = mapRuns("p", "e", [fixtureRun]);
    expect(result.updatedAt).toBe("2026-03-01T12:00:00Z");
  });

  it("falls back to created when finished is absent", () => {
    const [result] = mapRuns("p", "e", [fixtureRunPending]);
    expect(result.updatedAt).toBe(fixtureRunPending.created);
  });

  it("maps an empty array to an empty array", () => {
    expect(mapRuns("p", "e", [])).toEqual([]);
  });
});

describe("mapAssets", () => {
  it("maps id from asset.id", () => {
    const [result] = mapAssets([fixtureAsset]);
    expect(result.id).toBe("asset-001");
  });

  it("maps name from assetId", () => {
    const [result] = mapAssets([fixtureAsset]);
    expect(result.name).toBe("asset-001");
  });

  it("builds summary from type and format", () => {
    const [result] = mapAssets([fixtureAsset]);
    expect(result.summary).toBe("model • pickle");
  });

  it("maps sizeBytes", () => {
    const [result] = mapAssets([fixtureAsset]);
    expect(result.sizeBytes).toBe(1024);
  });

  it("forwards optional projectId", () => {
    const [withProject] = mapAssets([fixtureAsset], "proj-alpha");
    expect(withProject.projectId).toBe("proj-alpha");

    const [withoutProject] = mapAssets([fixtureAsset]);
    expect(withoutProject.projectId).toBeUndefined();
  });
});

describe("mapWorkflows", () => {
  it("constructs workflow id from experiment id", () => {
    const rawExp = [fixtureExperiment];
    const expSummaries = mapExperiments("proj-alpha", rawExp);
    const [result] = mapWorkflows(expSummaries, rawExp);
    expect(result.id).toBe("workflow:exp-001");
  });

  it("names workflow from experiment name", () => {
    const rawExp = [fixtureExperiment];
    const expSummaries = mapExperiments("proj-alpha", rawExp);
    const [result] = mapWorkflows(expSummaries, rawExp);
    expect(result.name).toBe("Baseline workflow");
  });

  it("sets summary to workflow path", () => {
    const rawExp = [fixtureExperiment];
    const expSummaries = mapExperiments("proj-alpha", rawExp);
    const [result] = mapWorkflows(expSummaries, rawExp);
    expect(result.summary).toBe("workflow.py");
  });

  it("maps an empty array to an empty array", () => {
    expect(mapWorkflows([], [])).toEqual([]);
  });
});

describe("mapWorkspaceTree", () => {
  it("sets id to 'workspace-root'", () => {
    const result = mapWorkspaceTree("/ws", {});
    expect(result.id).toBe("workspace-root");
  });

  it("uses response.path when present", () => {
    const result = mapWorkspaceTree("/fallback", { path: "/actual/path" });
    expect(result.name).toBe("/actual/path");
    expect(result.path).toBe("/actual/path");
  });

  it("falls back to rootPath when response.path is absent", () => {
    const result = mapWorkspaceTree("/fallback", {});
    expect(result.name).toBe("/fallback");
  });

  it("maps children recursively", () => {
    const result = mapWorkspaceTree("/ws", {
      children: [
        { name: "src", path: "/ws/src", type: "directory" },
        { name: "main.py", path: "/ws/main.py", type: "file" },
      ],
    });
    expect(result.children).toHaveLength(2);
    expect(result.children[0].kind).toBe("directory");
    expect(result.children[1].kind).toBe("file");
  });

  it("produces empty children array when response has none", () => {
    const result = mapWorkspaceTree("/ws", {});
    expect(result.children).toEqual([]);
  });
});

describe("mapAgentSessions", () => {
  const rawSession = {
    sessionId: "sess-abc",
    status: "completed",
    goalDescription: "Run baseline experiment",
    createdAt: "2026-03-01T10:00:00Z",
    events: [
      { type: "PlanCreatedEvent", ts: "2026-03-01T10:00:01Z", payload: {} },
      { type: "SessionCompletedEvent", ts: "2026-03-01T10:05:00Z", payload: {} },
    ],
  };

  it("maps sessionId to id", () => {
    const [result] = mapAgentSessions([rawSession]);
    expect(result.id).toBe("sess-abc");
  });

  it("maps goalDescription", () => {
    const [result] = mapAgentSessions([rawSession]);
    expect(result.goalDescription).toBe("Run baseline experiment");
  });

  it("maps status", () => {
    const [result] = mapAgentSessions([rawSession]);
    expect(result.status).toBe("completed");
  });

  it("maps createdAt", () => {
    const [result] = mapAgentSessions([rawSession]);
    expect(result.createdAt).toBe("2026-03-01T10:00:00Z");
  });

  it("counts events", () => {
    const [result] = mapAgentSessions([rawSession]);
    expect(result.eventCount).toBe(2);
  });

  it("maps an empty array to an empty array", () => {
    expect(mapAgentSessions([])).toEqual([]);
  });
});
