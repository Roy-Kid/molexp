export const EMPTY_COPY = {
  experiments: {
    title: "No experiments yet.",
    description: 'Click "Create Experiment" to start.',
  },
  runs: {
    title: "No runs yet.",
    description: 'Click "Start Run" to begin.',
  },
  projectRuns: {
    title: "No runs yet.",
    description: "No runs have been created for this project.",
  },
  assets: {
    title: "No assets yet.",
    description: "No assets have been recorded here.",
  },
  entries: {
    title: "No entries available.",
  },
  projectsFilter: {
    title: "No projects match the current filter.",
  },
  workspace: {
    title: "No workspace files loaded.",
  },
  emptyFolder: {
    title: "Empty folder",
  },
  agentSessions: {
    title: "No sessions yet.",
    description: "Start a new goal to create one.",
  },
} as const;
