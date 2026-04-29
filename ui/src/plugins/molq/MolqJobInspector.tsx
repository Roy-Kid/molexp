import { useEffect, useState } from "react";
import type { JSX } from "react";

import { StatusBadge } from "@/app/components/entity";
import { molqApi } from "@/plugins/molq/api";
import { MolqLogPanel } from "@/plugins/molq/MolqLogPanel";
import { formatDuration, formatTimestamp } from "@/plugins/molq/format";
import type { MolqJobDetail, MolqJobSummary } from "@/plugins/molq/types";

interface MolqJobInspectorProps {
  job: MolqJobSummary | null;
}

const Field = ({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}): JSX.Element => (
  <div className="flex justify-between gap-3 py-1">
    <span className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</span>
    <span className="min-w-0 truncate text-right font-mono text-xs text-foreground">{children}</span>
  </div>
);

export const MolqJobInspector = ({ job }: MolqJobInspectorProps): JSX.Element => {
  const [detail, setDetail] = useState<MolqJobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    setError(null);
    if (!job) return;

    let cancelled = false;
    molqApi
      .getJob(job.target, job.jobId)
      .then((next) => {
        if (!cancelled) setDetail(next);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [job]);

  if (!job) {
    return (
      <div className="flex h-full flex-col items-center justify-center border-l border-border/70 bg-muted/10 p-6 text-center text-xs text-muted-foreground">
        Select a job to inspect.
      </div>
    );
  }

  return (
    <div className="flex h-full min-w-0 flex-col border-l border-border/70 bg-background">
      <header className="border-b border-border/60 p-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="truncate text-sm font-semibold text-foreground">
            {job.name ?? job.jobId.slice(0, 12)}
          </h3>
          <StatusBadge status={job.state} size="sm" />
        </div>
        <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
          {job.schedulerJobId ?? job.jobId}
        </p>
      </header>

      <div className="space-y-4 overflow-auto p-3">
        <section className="space-y-1">
          <h4 className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            Details
          </h4>
          <div className="divide-y divide-border/40">
            <Field label="Target">{job.target}</Field>
            <Field label="Backend">{job.scheduler ?? "—"}</Field>
            <Field label="Cluster">{job.clusterName ?? "—"}</Field>
            <Field label="Submitted">{formatTimestamp(job.submittedAt)}</Field>
            <Field label="Started">{formatTimestamp(job.startedAt)}</Field>
            <Field label="Finished">{formatTimestamp(job.finishedAt)}</Field>
            <Field label="Duration">{formatDuration(job.durationSeconds)}</Field>
            <Field label="Exit code">{job.exitCode ?? "—"}</Field>
          </div>
        </section>

        {error && <div className="text-xs text-destructive">{error}</div>}

        {detail && (
          <>
            {detail.commandDisplay && (
              <section className="space-y-1">
                <h4 className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Command
                </h4>
                <pre className="break-all rounded border border-border/50 bg-muted/30 p-2 font-mono text-[11px] text-foreground">
                  {detail.commandDisplay}
                </pre>
              </section>
            )}

            {detail.failureReason && (
              <section className="space-y-1">
                <h4 className="text-[10px] font-semibold uppercase tracking-wide text-destructive">
                  Failure reason
                </h4>
                <p className="rounded border border-destructive/40 bg-destructive/10 p-2 text-[11px] text-destructive">
                  {detail.failureReason}
                </p>
              </section>
            )}

            <section className="space-y-1">
              <h4 className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                Recent transitions
              </h4>
              <ol className="space-y-1">
                {detail.transitions
                  .slice()
                  .reverse()
                  .slice(0, 8)
                  .map((t, idx) => (
                    <li
                      key={`${t.timestamp}-${idx}`}
                      className="flex justify-between gap-2 border-b border-border/30 py-1 text-[11px]"
                    >
                      <span className="font-mono text-muted-foreground">
                        {formatTimestamp(t.timestamp)}
                      </span>
                      <span className="font-mono">
                        {t.fromState ? `${t.fromState} → ` : ""}
                        {t.toState}
                      </span>
                    </li>
                  ))}
              </ol>
            </section>
          </>
        )}
      </div>

      <MolqLogPanel target={job.target} jobId={job.jobId} />
    </div>
  );
};
