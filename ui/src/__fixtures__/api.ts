/**
 * Shared fixture data for unit tests.
 * Equivalent role to Python's conftest.py — import from here, do not inline in test files.
 */

import type { AssetResponse } from "@/api/generated/models/AssetResponse";
import type { ExperimentResponse } from "@/api/generated/models/ExperimentResponse";
import type { ProjectResponse } from "@/api/generated/models/ProjectResponse";
import type { RunResponse } from "@/api/generated/models/RunResponse";
import type { ExperimentSummary, ProjectSummary, RunSummary } from "@/app/types";

export const fixtureProject: ProjectResponse = {
  id: "proj-alpha",
  projectId: "proj-alpha",
  name: "Alpha Project",
  created: "2026-03-01T00:00:00Z",
  description: "First project",
};

export const fixtureProjectNoDescription: ProjectResponse = {
  id: "proj-beta",
  projectId: "proj-beta",
  name: "Beta Project",
  created: "2026-03-02T00:00:00Z",
};

export const fixtureExperiment: ExperimentResponse = {
  id: "exp-001",
  experimentId: "exp-001",
  projectId: "proj-alpha",
  name: "Baseline",
  created: "2026-03-01T10:00:00Z",
  workflow: "workflow.py",
  description: "Baseline experiment",
};

export const fixtureExperimentNoDescription: ExperimentResponse = {
  id: "exp-002",
  experimentId: "exp-002",
  projectId: "proj-alpha",
  name: "Variant",
  created: "2026-03-02T10:00:00Z",
  workflow: "variant.py",
};

export const fixtureRun: RunResponse = {
  id: "run-abc",
  runId: "run-abc",
  projectId: "proj-alpha",
  experimentId: "exp-001",
  status: "succeeded",
  created: "2026-03-01T11:00:00Z",
  finished: "2026-03-01T12:00:00Z",
  parameters: { lr: 0.001 },
};

export const fixtureRunPending: RunResponse = {
  id: "run-def",
  runId: "run-def",
  projectId: "proj-alpha",
  experimentId: "exp-001",
  status: "pending",
  created: "2026-03-01T13:00:00Z",
};

export const fixtureRunFailed: RunResponse = {
  id: "run-ghi",
  runId: "run-ghi",
  projectId: "proj-alpha",
  experimentId: "exp-001",
  status: "failed",
  created: "2026-03-01T14:00:00Z",
};

export const fixtureRunCancelled: RunResponse = {
  id: "run-jkl",
  runId: "run-jkl",
  projectId: "proj-alpha",
  experimentId: "exp-001",
  status: "cancelled",
  created: "2026-03-01T15:00:00Z",
};

export const fixtureAsset: AssetResponse = {
  id: "asset-001",
  assetId: "asset-001",
  type: "model",
  format: "pickle",
  created: "2026-03-01T16:00:00Z",
  size: 1024,
  contentHash: "abc123def456",
};

export const fixtureProjectSummary: ProjectSummary = {
  id: "proj-alpha",
  name: "Alpha Project",
  status: "active",
  summary: "First project",
  updatedAt: "2026-03-01T00:00:00Z",
};

export const fixtureExperimentSummary: ExperimentSummary = {
  id: "exp-001",
  name: "Baseline",
  status: "active",
  summary: "Baseline experiment",
  workflowFile: "workflow.py",
  updatedAt: "2026-03-01T10:00:00Z",
  projectId: "proj-alpha",
};

export const fixtureRunSummary: RunSummary = {
  id: "run-abc",
  name: "run-abc",
  status: "succeeded",
  summary: "Status: succeeded",
  updatedAt: "2026-03-01T12:00:00Z",
  projectId: "proj-alpha",
  experimentId: "exp-001",
  executorInfo: { backend: "local" },
};
