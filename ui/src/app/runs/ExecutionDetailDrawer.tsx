import { X } from "lucide-react";
import type { JSX, ReactNode } from "react";
import { StatusBadge } from "@/app/components/entity";
import { listExecutionDetails } from "@/app/registry";
import { formatDuration, formatTimestamp } from "@/lib/format-time";

import type { WorkspaceExecutionRow, WorkspaceRunRow } from "./types";

interface ExecutionDetailDrawerProps {
  run: WorkspaceRunRow;
  execution: WorkspaceExecutionRow;
  onClose: () => void;
}

export const ExecutionDetailDrawer = ({
  run,
  execution,
  onClose,
}: ExecutionDetailDrawerProps): JSX.Element => {
  const details = listExecutionDetails(execution.backend);

  return (
    <aside className="flex h-full w-[400px] flex-col border-l border-border bg-card">
      <header className="flex items-start justify-between gap-3 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <p className="truncate text-xs uppercase text-muted-foreground">Execution</p>
          <p className="truncate font-mono text-sm text-foreground" title={execution.executionId}>
            {execution.executionId}
          </p>
          <div className="mt-1 flex items-center gap-2">
            <StatusBadge status={execution.status} size="sm" />
            <span className="text-xs text-muted-foreground">
              of run <span className="font-mono">{run.name}</span>
            </span>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-muted-foreground hover:bg-accent"
          aria-label="Close detail panel"
        >
          <X className="h-4 w-4" />
        </button>
      </header>

      <div className="flex-1 overflow-y-auto">
        <Section title="Summary">
          <Field label="Backend" value={execution.backend ?? "—"} />
          <Field label="Started" value={formatTimestamp(execution.startedAt)} />
          <Field label="Finished" value={formatTimestamp(execution.finishedAt)} />
          <Field label="Duration" value={formatDuration(execution.durationSeconds)} />
          {execution.schedulerJobId && (
            <Field label="Scheduler job" value={execution.schedulerJobId} mono />
          )}
        </Section>

        {Object.keys(execution.backendMetadata).length > 0 && (
          <Section title="Backend metadata">
            {Object.entries(execution.backendMetadata).map(([key, value]) => (
              <Field key={key} label={key} value={value} mono />
            ))}
          </Section>
        )}

        {details.map((entry) => {
          const Component = entry.Component;
          return (
            <Section key={entry.id} title={entry.title}>
              <Component execution={execution} runId={run.id} />
            </Section>
          );
        })}
      </div>
    </aside>
  );
};

const Section = ({ title, children }: { title: string; children: ReactNode }): JSX.Element => (
  <section className="border-b border-border/60 px-4 py-3">
    <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
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
