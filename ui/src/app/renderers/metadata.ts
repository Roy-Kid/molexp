import type {
  AgentSessionSummary,
  AssetSummary,
  ExperimentSummary,
  ProjectSummary,
  RunSummary,
  Selection,
  SemanticObjectType,
  WorkflowSummary,
  WorkspaceSnapshot,
} from "@/app/types";

export interface MetadataField {
  label: string;
  value: string;
}

const findProject = (snapshot: WorkspaceSnapshot, id: string): ProjectSummary | null => {
  return snapshot.projects.find((project) => project.id === id) ?? null;
};

const findExperiment = (snapshot: WorkspaceSnapshot, id: string): ExperimentSummary | null => {
  return snapshot.experiments.find((experiment) => experiment.id === id) ?? null;
};

const findRun = (snapshot: WorkspaceSnapshot, id: string): RunSummary | null => {
  return snapshot.runs.find((run) => run.id === id) ?? null;
};

const findAsset = (snapshot: WorkspaceSnapshot, id: string): AssetSummary | null => {
  return snapshot.assets.find((asset) => asset.id === id) ?? null;
};

const findAgentSession = (snapshot: WorkspaceSnapshot, id: string): AgentSessionSummary | null => {
  return snapshot.agentSessions.find((s) => s.id === id) ?? null;
};

const findWorkflow = (snapshot: WorkspaceSnapshot, id: string): WorkflowSummary | null => {
  return snapshot.workflows.find((workflow) => workflow.id === id) ?? null;
};

const emptyFields = (objectType: SemanticObjectType, objectId: string): MetadataField[] => {
  return [
    { label: "Object Type", value: objectType },
    { label: "Object ID", value: objectId },
    { label: "Status", value: "Missing in snapshot" },
  ];
};

export const buildMetadataFields = (
  selection: Selection,
  snapshot: WorkspaceSnapshot,
): MetadataField[] => {
  const lookupByType: Record<SemanticObjectType, () => MetadataField[]> = {
    project: () => {
      const project = findProject(snapshot, selection.objectId);
      if (!project) {
        return emptyFields("project", selection.objectId);
      }
      return [
        { label: "Project", value: project.name },
        { label: "Status", value: project.status },
        { label: "Summary", value: project.summary },
        { label: "Updated", value: project.updatedAt },
      ];
    },
    experiment: () => {
      const experiment = findExperiment(snapshot, selection.objectId);
      if (!experiment) {
        return emptyFields("experiment", selection.objectId);
      }
      return [
        { label: "Experiment", value: experiment.name },
        { label: "Project", value: experiment.projectId },
        { label: "Status", value: experiment.status },
        { label: "Summary", value: experiment.summary },
        { label: "Updated", value: experiment.updatedAt },
      ];
    },
    run: () => {
      const run = findRun(snapshot, selection.objectId);
      if (!run) {
        return emptyFields("run", selection.objectId);
      }
      const fields: MetadataField[] = [
        { label: "Run", value: run.name },
        { label: "Project", value: run.projectId },
        { label: "Experiment", value: run.experimentId },
        { label: "Status", value: run.status },
        { label: "Summary", value: run.summary },
        { label: "Updated", value: run.updatedAt },
      ];
      if (run.executorInfo.backend) {
        fields.push({ label: "Backend", value: run.executorInfo.backend });
      }
      if (run.executorInfo.scheduler) {
        fields.push({ label: "Scheduler", value: run.executorInfo.scheduler });
      }
      if (run.executorInfo.job_id) {
        fields.push({ label: "Job ID", value: run.executorInfo.job_id });
      }
      return fields;
    },
    asset: () => {
      const asset = findAsset(snapshot, selection.objectId);
      if (!asset) {
        return emptyFields("asset", selection.objectId);
      }
      return [
        { label: "Asset", value: asset.name },
        { label: "Status", value: asset.status },
        { label: "Summary", value: asset.summary },
        { label: "Updated", value: asset.updatedAt },
        { label: "Size", value: `${asset.sizeBytes} bytes` },
      ];
    },
    workflow: () => {
      const workflow = findWorkflow(snapshot, selection.objectId);
      if (!workflow) {
        return emptyFields("workflow", selection.objectId);
      }
      return [
        { label: "Workflow", value: workflow.name },
        { label: "Project", value: workflow.projectId },
        { label: "Experiment", value: workflow.experimentId },
        { label: "Status", value: workflow.status },
        { label: "Summary", value: workflow.summary },
        { label: "Updated", value: workflow.updatedAt },
      ];
    },
    agent: () => {
      const session = findAgentSession(snapshot, selection.objectId);
      if (!session) {
        return emptyFields("agent", selection.objectId);
      }
      return [
        { label: "Session", value: session.id },
        { label: "Status", value: session.status },
        { label: "Goal", value: session.goalDescription },
        { label: "Events", value: String(session.eventCount) },
        { label: "Created", value: session.createdAt },
      ];
    },
    "workspace-file": () => {
      if (selection.objectType !== "workspace-file") {
        return emptyFields("workspace-file", selection.objectId);
      }
      return [
        { label: "File", value: selection.filePath },
        { label: "Kind", value: selection.fileKind },
      ];
    },
  };

  const buildFields = lookupByType[selection.objectType];
  return buildFields();
};
