import type { JSX, ReactNode } from "react";

import { StatusBadge } from "@/app/components/entity";
import { formatDuration, formatRelative, formatTimestamp } from "@/lib/format-time";

import { RunsRecentEvents } from "../RunsRecentEvents";
import type { WorkspaceExecutionRow, WorkspaceRunRow } from "../types";

interface RunInspectorDetailsProps {
  run: WorkspaceRunRow;
  selectedExecutionId: string | null;
  onSelectExecution: (id: string | null) => void;
}

const computeRunDuration = (run: WorkspaceRunRow): number | null => {
  const start = run.executions
    .map((exec) => (exec.startedAt ? new Date(exec.startedAt).getTime() : NaN))
    .filter((v) => !Number.isNaN(v))
    .sort((a, b) => a - b)[0];
  if (typeof start !== "number") return null;
  const end = run.finishedAt ? new Date(run.finishedAt).getTime() : Date.now();
  if (Number.isNaN(end)) return null;
  return Math.max(0, (end - start) / 1000);
};

export const RunInspectorDetails = ({
  run,
  selectedExecutionId,
  onSelectExecution,
}: RunInspectorDetailsProps): JSX.Element => {
  const parameterEntries = Object.entries(run.parameters);
  const duration = computeRunDuration(run);

  return (
    <div className="flex flex-col">
      <Section title="Backend">
        <Field label="Backend" value={run.backend ?? "—"} />
        <Field label="Cluster" value={run.cluster ?? "—"} />
        <Field label="Scheduler" value={run.scheduler ?? "—"} />
        <Field label="Profile" value={run.profile ?? "—"} />
        {run.latestSchedulerJobId && (
          <Field label="Scheduler job" value={run.latestSchedulerJobId} mono />
        )}
      </Section>

      <Section title="Lifecycle">
        <Field label="Submitted" value={formatTimestamp(run.createdAt)} />
        <Field
          label="Started"
          value={run.executions[0]?.startedAt ? formatTimestamp(run.executions[0].startedAt) : "—"}
        />
        <Field label="Finished" value={formatTimestamp(run.finishedAt)} />
        <Field label="Duration" value={formatDuration(duration)} />
        <Field label="Executions" value={String(run.executionCount)} />
      </Section>

      <Section title={`Recent events (${Math.min(8, 1 + 2 * run.executions.length)})`}>
        <RunsRecentEvents run={run} />
      </Section>

      {run.executions.length > 0 && (
        <Section title={`Attempts (${run.executions.length})`}>
          <ul className="space-y-1">
            {run.executions.map((execution) => (
              <ExecutionRow
                key={execution.executionId}
                execution={execution}
                selected={execution.executionId === selectedExecutionId}
                onSelect={() =>
                  onSelectExecution(
                    execution.executionId === selectedExecutionId ? null : execution.executionId,
                  )
                }
              />
            ))}
          </ul>
        </Section>
      )}

      {parameterEntries.length > 0 && (
        <Section title="Parameters">
          {parameterEntries.map(([key, value]) => (
            <Field key={key} label={key} value={String(value)} mono />
          ))}
        </Section>
      )}
    </div>
  );
};

const Section = ({ title, children }: { title: string; children: ReactNode }): JSX.Element => (
  <section className="border-b border-border/60 px-4 py-3">
    <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
      {title}
    </h3>
    <div className="space-y-1.5 text-xs">{children}</div>
  </section>
);

const Field = ({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}): JSX.Element => (
  <div className="flex items-baseline justify-between gap-3">
    <span className="text-muted-foreground">{label}</span>
    <span
      className={
        mono
          ? "max-w-[60%] truncate font-mono text-foreground"
          : "max-w-[60%] truncate text-foreground"
      }
      title={value}
    >
      {value}
    </span>
  </div>
);

interface ExecutionRowProps {
  execution: WorkspaceExecutionRow;
  selected: boolean;
  onSelect: () => void;
}

const ExecutionRow = ({ execution, selected, onSelect }: ExecutionRowProps): JSX.Element => (
  <li>
    <button
      type="button"
      onClick={onSelect}
      className={
        selected
          ? "flex w-full items-center justify-between gap-2 rounded border border-info/40 bg-info-soft px-2 py-1.5 text-left text-xs"
          : "flex w-full items-center justify-between gap-2 rounded border border-border/60 bg-background px-2 py-1.5 text-left text-xs hover:bg-accent/40"
      }
    >
      <span className="truncate font-mono text-muted-foreground">
        {execution.executionId.slice(0, 14)}
      </span>
      <div className="flex items-center gap-2">
        <StatusBadge status={execution.status} size="sm" dot />
        <span className="text-muted-foreground">{formatRelative(execution.startedAt)}</span>
      </div>
    </button>
  </li>
);
