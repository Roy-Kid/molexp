export interface MolqTargetSummary {
  name: string;
  scheduler: string;
  clusterName: string | null;
  jobsDir: string | null;
  healthy: boolean;
  healthReason: string | null;
  activeJobs: number;
}

export interface MolqJobSummary {
  target: string;
  jobId: string;
  schedulerJobId: string | null;
  clusterName: string | null;
  scheduler: string | null;
  name: string | null;
  state: string;
  submittedAt: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  exitCode: number | null;
  durationSeconds: number | null;
  cwd: string | null;
}

export interface MolqQueueStats {
  running: number;
  pending: number;
  failed: number;
  succeeded: number;
  avgWaitSeconds: number | null;
}

export interface MolqJobsResponse {
  jobs: MolqJobSummary[];
  stats: MolqQueueStats;
  total: number;
}

export interface MolqJobTransition {
  timestamp: string;
  fromState: string | null;
  toState: string;
  reason: string | null;
}

export interface MolqJobDetail extends MolqJobSummary {
  failureReason: string | null;
  metadata: Record<string, string>;
  commandDisplay: string | null;
  transitions: MolqJobTransition[];
  upstreamTotal: number;
  upstreamSatisfied: number;
  downstreamTotal: number;
}
