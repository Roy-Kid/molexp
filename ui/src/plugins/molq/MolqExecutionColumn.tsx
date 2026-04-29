import type { JSX } from "react";

import type { ExecutionColumnRenderProps } from "@/plugins/types";

const Empty = (): JSX.Element => <span className="text-muted-foreground/60">—</span>;

const Cluster = ({ execution }: ExecutionColumnRenderProps): JSX.Element => {
  const cluster = execution.metadata.cluster_name;
  const scheduler = execution.metadata.scheduler;
  if (!cluster && !scheduler) return <Empty />;
  return (
    <span className="font-mono text-[11px]">
      {cluster ?? "—"}
      {scheduler && (
        <span className="ml-1 text-muted-foreground">[{scheduler}]</span>
      )}
    </span>
  );
};

const SchedulerJob = ({ execution }: ExecutionColumnRenderProps): JSX.Element => {
  const id = execution.schedulerJobId ?? execution.metadata.scheduler_job_id;
  if (!id) return <Empty />;
  return <span className="font-mono text-[11px]">{id}</span>;
};

export const MolqExecutionColumn = {
  Cluster,
  SchedulerJob,
};
