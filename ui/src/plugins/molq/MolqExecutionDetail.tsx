import type { JSX } from "react";

import type { ExecutionDetailRenderProps } from "@/plugins/types";

/**
 * Detail section rendered inside the workspace runs drawer for executions
 * whose backend is "molq". Currently surfaces the captured submission
 * metadata; a future iteration can add live transitions / log tail once
 * the molq profile name is plumbed into executor_info.
 */
export const MolqExecutionDetail = ({ execution }: ExecutionDetailRenderProps): JSX.Element => {
  const entries = Object.entries(execution.metadata).filter(
    ([key]) => key !== "scheduler_job_id" && key !== "backend",
  );

  if (entries.length === 0 && !execution.schedulerJobId) {
    return (
      <p className="text-xs text-muted-foreground">
        No molq submission metadata captured for this execution.
      </p>
    );
  }

  return (
    <dl className="space-y-1.5 text-xs">
      {execution.schedulerJobId && (
        <Row label="scheduler_job_id" value={execution.schedulerJobId} />
      )}
      {entries.map(([key, value]) => (
        <Row key={key} label={key} value={value} />
      ))}
    </dl>
  );
};

const Row = ({ label, value }: { label: string; value: string }): JSX.Element => (
  <div className="flex items-baseline justify-between gap-3">
    <dt className="text-muted-foreground">{label}</dt>
    <dd className="max-w-[60%] truncate font-mono text-foreground" title={value}>
      {value}
    </dd>
  </div>
);
