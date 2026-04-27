import { ExternalLink, X } from "lucide-react";
import type { JSX, ReactNode } from "react";

import { StatusBadge } from "@/app/components/entity";
import { Button } from "@/components/ui/button";
import { formatRelative, formatTimestamp } from "@/lib/format-time";

import type { WorkspaceExecutionRow, WorkspaceRunRow } from "./types";

interface RunDetailDrawerProps {
  run: WorkspaceRunRow;
  onClose: () => void;
  onOpenRun: () => void;
  onSelectExecution: (execution: WorkspaceExecutionRow) => void;
}

export const RunDetailDrawer = ({
  run,
  onClose,
  onOpenRun,
  onSelectExecution,
}: RunDetailDrawerProps): JSX.Element => {
  const backendLabel = run.backend
    ? run.cluster
      ? `${run.backend} · ${run.cluster}`
      : run.backend
    : "—";

  const parameterEntries = Object.entries(run.parameters);

  return (
    <aside className="flex h-full w-[400px] flex-col border-l border-border bg-card">
      <header className="flex items-start justify-between gap-3 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <p className="truncate text-xs uppercase text-muted-foreground">Run</p>
          <p className="truncate font-mono text-sm text-foreground" title={run.id}>
            {run.name || run.id}
          </p>
          <div className="mt-1 flex items-center gap-2">
            <StatusBadge status={run.status} size="sm" />
            <span className="text-xs text-muted-foreground">
              {run.projectName} · {run.experimentName}
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
          <Field label="Backend" value={backendLabel} />
          <Field label="Profile" value={run.profile ?? "—"} />
          <Field label="Created" value={formatTimestamp(run.createdAt)} />
          <Field label="Finished" value={formatTimestamp(run.finishedAt)} />
          <Field label="Executions" value={String(run.executionCount)} />
          {run.latestSchedulerJobId && (
            <Field label="Latest scheduler job" value={run.latestSchedulerJobId} mono />
          )}
        </Section>

        {parameterEntries.length > 0 && (
          <Section title="Parameters">
            {parameterEntries.map(([key, value]) => (
              <Field key={key} label={key} value={String(value)} mono />
            ))}
          </Section>
        )}

        <Section title={`Executions (${run.executions.length})`}>
          {run.executions.length === 0 ? (
            <p className="italic text-muted-foreground">No execution attempts yet.</p>
          ) : (
            <ul className="space-y-1.5">
              {run.executions.map((execution) => (
                <li key={execution.executionId}>
                  <button
                    type="button"
                    onClick={() => onSelectExecution(execution)}
                    className="flex w-full items-center justify-between gap-2 rounded border border-border/60 bg-background px-2 py-1.5 text-left text-xs hover:bg-accent/50"
                  >
                    <span className="truncate font-mono text-muted-foreground">
                      {execution.executionId.slice(0, 14)}
                    </span>
                    <div className="flex items-center gap-2">
                      <StatusBadge status={execution.status} size="sm" />
                      <span className="text-muted-foreground">
                        {formatRelative(execution.startedAt)}
                      </span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Section>
      </div>

      <footer className="border-t border-border px-4 py-3">
        <Button size="sm" variant="outline" onClick={onOpenRun} className="w-full">
          <ExternalLink className="mr-1.5 h-3.5 w-3.5" />
          Open run detail
        </Button>
      </footer>
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
