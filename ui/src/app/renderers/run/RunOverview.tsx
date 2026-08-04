import { Activity, Braces, GitBranch, type LucideIcon, PackageOpen } from "lucide-react";
import type { ReactNode } from "react";
import { CopyButton } from "@/app/components/entity";
// Module-path import (not the barrel) — see KnowledgeBacklinksCard loader note.
import { KnowledgeBacklinksCard } from "@/app/components/entity/KnowledgeBacklinksCard";
import { formatScalar } from "@/app/renderers/dashboardData";
import type {
  ApiAssetResponse,
  ExperimentSummary,
  ProjectSummary,
  RunSummary,
  WorkflowSummary,
} from "@/app/types";
import { WorkbenchAction } from "@/components/workbench";
import { formatDateTime } from "@/lib/datetime";
import { cn } from "@/lib/utils";

interface RunOverviewProps {
  run: RunSummary;
  project?: ProjectSummary;
  experiment?: ExperimentSummary;
  workflow?: WorkflowSummary;
  backend: string;
  duration: string | null;
  attemptCount: number;
  assets: ApiAssetResponse[];
  parameters: [string, unknown][];
  results: [string, unknown][];
  onOpenProject: () => void;
  onOpenExperiment: () => void;
  onOpenWorkflow?: () => void;
}

interface SectionProps {
  icon: LucideIcon;
  title: string;
  count?: number;
  children: ReactNode;
  className?: string;
  copyText?: string;
  copyLabel?: string;
}

const Section = ({
  icon: Icon,
  title,
  count,
  children,
  className,
  copyText,
  copyLabel,
}: SectionProps): JSX.Element => (
  <section className={cn("relative min-w-0", className)} aria-label={title}>
    <header className="flex h-control-comfortable items-center gap-2 border-b border-border/70 bg-surface-subtle/60 px-3 after:absolute after:left-3 after:top-0 after:h-px after:w-10 after:bg-gradient-to-r after:from-accent after:to-transparent">
      <Icon className="size-3.5 text-accent" aria-hidden />
      <h3 className="font-mono text-label font-medium uppercase tracking-wider text-foreground">
        {title}
      </h3>
      <div className="ml-auto flex items-center gap-1">
        {count !== undefined && (
          <span className="rounded-control bg-accent-muted px-2 py-0.5 font-mono text-micro tabular-nums text-accent-muted-foreground">
            {count}
          </span>
        )}
        {copyText !== undefined && (
          <CopyButton value={copyText} label={copyLabel ?? title} className="size-5" />
        )}
      </div>
    </header>
    {children}
  </section>
);

interface FactProps {
  label: string;
  value: ReactNode;
  mono?: boolean;
  title?: string;
  copyValue?: string;
}

const Fact = ({ label, value, mono = false, title, copyValue }: FactProps): JSX.Element => (
  <div className="min-w-0 border-b border-border/60 px-3 py-1.5 lg:border-r last:lg:border-r-0">
    <dt className="text-micro uppercase tracking-wide text-muted-foreground">{label}</dt>
    <dd
      className={cn(
        "mt-0.5 flex min-w-0 items-center gap-1 text-body font-medium text-foreground",
        mono && "font-mono text-label font-normal",
      )}
      title={title}
    >
      <span className="min-w-0 truncate">{value}</span>
      {copyValue !== undefined && <CopyButton value={copyValue} label={label} className="size-5" />}
    </dd>
  </div>
);

interface EntryGridProps {
  entries: [string, unknown][];
  empty: string;
}

const EntryGrid = ({ entries, empty }: EntryGridProps): JSX.Element => {
  if (entries.length === 0) {
    return <p className="px-3 py-3 text-label text-muted-foreground">{empty}</p>;
  }

  return (
    <dl className="grid sm:grid-cols-2">
      {entries.map(([key, rawValue]) => {
        const value = formatScalar(rawValue);
        return (
          <div
            key={key}
            className="grid min-w-0 grid-cols-(--run-overview-grid-columns) items-center gap-1 border-b border-border/60 px-3 py-1.5 odd:sm:border-r hover:bg-interactive/40"
          >
            <dt className="truncate text-label text-muted-foreground" title={key}>
              {key}
            </dt>
            <dd className="truncate text-right font-mono text-label text-foreground" title={value}>
              {value}
            </dd>
            <CopyButton value={value} label={key} className="size-5" />
          </div>
        );
      })}
    </dl>
  );
};

interface RelationProps {
  label: string;
  value: string;
  mono?: boolean;
  onClick?: () => void;
  copyValue?: string;
}

const Relation = ({
  label,
  value,
  mono = false,
  onClick,
  copyValue,
}: RelationProps): JSX.Element => {
  const valueClass = cn(
    "block min-w-0 truncate text-right text-label text-foreground",
    mono && "font-mono text-micro",
  );

  return (
    <div className="flex min-h-control items-center gap-2 border-b border-border/60 px-3 py-1 last:border-b-0 hover:bg-interactive/40">
      <dt className="flex-none text-micro uppercase tracking-wide text-muted-foreground">
        {label}
      </dt>
      <dd className="ml-auto min-w-0">
        {onClick ? (
          <WorkbenchAction
            kind="ghost"
            size="content"
            type="button"
            className={cn(
              valueClass,
              "rounded-control px-1 hover:bg-interactive hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            )}
            onClick={onClick}
            title={value}
          >
            {value}
          </WorkbenchAction>
        ) : (
          <span className={valueClass} title={value}>
            {value}
          </span>
        )}
      </dd>
      {copyValue !== undefined && <CopyButton value={copyValue} label={label} className="size-5" />}
    </div>
  );
};

export const RunOverview = ({
  run,
  project,
  experiment,
  workflow,
  backend,
  duration,
  attemptCount,
  assets,
  parameters,
  results,
  onOpenProject,
  onOpenExperiment,
  onOpenWorkflow,
}: RunOverviewProps): JSX.Element => {
  const coordinates = JSON.stringify(
    { projectId: run.projectId, experimentId: run.experimentId, runId: run.id },
    null,
    2,
  );
  const parameterJson = JSON.stringify(run.parameters ?? {}, null, 2);
  const resultJson = JSON.stringify(run.results ?? {}, null, 2);

  return (
    <div className="molexp-dashboard flex-1 overflow-auto bg-canvas p-3">
      <div className="mx-auto grid min-h-full w-full max-w-7xl overflow-hidden bg-surface/95 xl:grid-cols-(--run-layout-columns)">
        <main className="min-w-0">
          {run.errorMessage && (
            <section
              className="relative border-b border-status-failed/30 bg-status-failed-soft px-3 py-2 pr-10"
              aria-label="Run error"
            >
              <p className="text-label font-medium text-status-failed-foreground">Run error</p>
              <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-label leading-relaxed text-status-failed-foreground">
                {run.errorMessage}
              </pre>
              <CopyButton
                value={run.errorMessage}
                label="run error"
                className="absolute right-3 top-2"
              />
            </section>
          )}

          <Section
            icon={Activity}
            title="Execution"
            copyText={coordinates}
            copyLabel="run coordinates"
          >
            {run.summary && (
              <p className="border-b border-l border-border/60 border-l-accent bg-accent-muted/30 px-3 py-2 text-body text-foreground">
                {run.summary}
              </p>
            )}
            <dl className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 lg:divide-x lg:divide-border/60">
              <Fact
                label="Started"
                value={formatDateTime(run.startedAt)}
                title={run.startedAt ?? undefined}
                copyValue={run.startedAt ?? undefined}
              />
              <Fact
                label="Finished"
                value={formatDateTime(run.finishedAt)}
                title={run.finishedAt ?? undefined}
                copyValue={run.finishedAt ?? undefined}
              />
              <Fact label="Duration" value={duration ?? "—"} />
              <Fact label="Backend" value={backend} mono copyValue={backend} />
              <Fact
                label="Profile"
                value={run.profile ?? "default"}
                mono
                copyValue={run.profile ?? "default"}
              />
              <Fact label="Attempts" value={attemptCount} />
              <Fact label="Assets" value={assets.length} />
              <Fact label="Results" value={results.length} />
            </dl>
          </Section>

          <div className="grid border-t border-border/70 lg:grid-cols-2 lg:divide-x lg:divide-border/70">
            <Section
              icon={Braces}
              title="Parameters"
              count={parameters.length}
              copyText={parameterJson}
              copyLabel="all parameters"
            >
              <EntryGrid entries={parameters} empty="No parameters on this run." />
            </Section>
            <Section
              icon={PackageOpen}
              title="Results"
              count={results.length}
              copyText={resultJson}
              copyLabel="all results"
            >
              <EntryGrid entries={results} empty="No results recorded yet." />
            </Section>
          </div>
        </main>

        <aside className="min-w-0 border-t border-border/70 bg-surface-subtle/45 xl:border-l xl:border-t-0">
          <Section
            icon={GitBranch}
            title="Lineage"
            copyText={coordinates}
            copyLabel="lineage coordinates"
          >
            <dl>
              <Relation
                label="Project"
                value={project?.name ?? run.projectId}
                onClick={onOpenProject}
                copyValue={run.projectId}
              />
              <Relation
                label="Experiment"
                value={experiment?.name ?? run.experimentId}
                onClick={onOpenExperiment}
                copyValue={run.experimentId}
              />
              {workflow && onOpenWorkflow && (
                <Relation
                  label="Workflow"
                  value={workflow.name || workflow.id}
                  onClick={onOpenWorkflow}
                  copyValue={workflow.id}
                />
              )}
              <Relation label="Run ID" value={run.id} mono copyValue={run.id} />
              <Relation
                label="Config"
                value={run.configHash ?? "—"}
                mono
                copyValue={run.configHash ?? undefined}
              />
            </dl>
          </Section>

          <div className="border-t border-border/70 p-3">
            <KnowledgeBacklinksCard
              kind="run"
              projectId={run.projectId}
              experimentId={run.experimentId}
              runId={run.id}
            />
          </div>
        </aside>
      </div>
    </div>
  );
};
