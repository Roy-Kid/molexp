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
  name: "Alpha Project",
  created: "2026-03-01T00:00:00Z",
  description: "First project",
};

export const fixtureProjectNoDescription: ProjectResponse = {
  id: "proj-beta",
  name: "Beta Project",
  created: "2026-03-02T00:00:00Z",
};

export const fixtureExperiment: ExperimentResponse = {
  id: "exp-001",
  projectId: "proj-alpha",
  name: "Baseline",
  created: "2026-03-01T10:00:00Z",
  workflow: "workflow.py",
  description: "Baseline experiment",
};

export const fixtureExperimentNoDescription: ExperimentResponse = {
  id: "exp-002",
  projectId: "proj-alpha",
  name: "Variant",
  created: "2026-03-02T10:00:00Z",
  workflow: "variant.py",
};

export const fixtureRun: RunResponse = {
  id: "run-abc",
  projectId: "proj-alpha",
  experimentId: "exp-001",
  status: "succeeded",
  created: "2026-03-01T11:00:00Z",
  finished: "2026-03-01T12:00:00Z",
  parameters: { lr: 0.001 },
};

export const fixtureRunPending: RunResponse = {
  id: "run-def",
  projectId: "proj-alpha",
  experimentId: "exp-001",
  status: "pending",
  created: "2026-03-01T13:00:00Z",
};

export const fixtureRunFailed: RunResponse = {
  id: "run-ghi",
  projectId: "proj-alpha",
  experimentId: "exp-001",
  status: "failed",
  created: "2026-03-01T14:00:00Z",
};

export const fixtureRunCancelled: RunResponse = {
  id: "run-jkl",
  projectId: "proj-alpha",
  experimentId: "exp-001",
  status: "cancelled",
  created: "2026-03-01T15:00:00Z",
};

export const fixtureAsset: AssetResponse = {
  id: "asset-001",
  name: "checkpoint.pt",
  kind: "artifact",
  scope_kind: "run",
  scope_ids: ["proj-alpha", "exp-001", "run-abc"],
  path: "artifacts/checkpoint.pt",
  created_at: "2026-03-01T16:00:00Z",
  updated_at: "2026-03-01T16:00:00Z",
  producer: {
    run_id: "run-abc",
    execution_id: "exec-001",
    task_id: null,
  },
  tags: {},
  extra: {
    mime: "application/octet-stream",
    size: 1024,
  },
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
  profile: null,
  configHash: null,
};
