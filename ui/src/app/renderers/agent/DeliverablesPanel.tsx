import { Check, ClipboardCopy, FileQuestion, Loader2, Package } from "lucide-react";
import { type JSX, useCallback, useEffect, useMemo, useState } from "react";
import type { PlanDetailResponse } from "@/api/generated/models/PlanDetailResponse";
import { StatusBadge } from "@/app/components/entity";
import type { PlanRef } from "@/app/renderers/agentEvents";
import { collectArtifacts, derivePlanRef } from "@/app/renderers/agentEvents";
import { workspaceApi } from "@/app/state/api";
import type { ApiSessionEvent, SemanticStatus } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { MarkdownContent } from "@/components/ui/markdown";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ArtifactBody } from "./artifacts";
import { planStage } from "./planStages";

// ---------------------------------------------------------------------------
// Deliverables panel — the right half of the agent session view.
//
// Pulls the agent's *products* out of the conversation so they read as
// reviewable documents, not chat noise:
//   • PlanMode session → one deliverable per the nine pipeline steps (proposal,
//     spec YAML, capabilities, IR topology, source, input set, dry-run,
//     execution report), fetched structurally from `GET /plans/{runId}`. The
//     left progress rail selects which step's deliverable shows here.
//   • Chat session with inline artifacts → an Artifacts list (plots/tables).
// `hasDeliverables` (below) decides whether the parent shows this panel at all.
// ---------------------------------------------------------------------------

/** True when a session has products worth a dedicated panel. */
export const hasDeliverables = (events: ApiSessionEvent[]): boolean =>
  derivePlanRef(events) !== null || collectArtifacts(events).length > 0;

const CopyButton = ({ text, label = "Copy" }: { text: string; label?: string }): JSX.Element => {
  const [copied, setCopied] = useState(false);
  const onCopy = useCallback(() => {
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  }, [text]);
  return (
    <button
      type="button"
      onClick={onCopy}
      className="inline-flex items-center gap-1 rounded-md border border-border/60 bg-card px-2 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
    >
      {copied ? <Check className="h-3 w-3 text-success" /> : <ClipboardCopy className="h-3 w-3" />}
      {copied ? "Copied" : label}
    </button>
  );
};

const PanelSection = ({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}): JSX.Element => (
  <section className="space-y-1.5">
    <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</h3>
    {children}
  </section>
);

// Spec field order mirrors the server's experiment-report renderer so the spec
// reads the same everywhere it's surfaced.
const SPEC_FIELDS: [string, string][] = [
  ["objective", "Objective"],
  ["background", "Background"],
  ["system_description", "System description"],
  ["scientific_hypothesis", "Scientific hypothesis"],
  ["experimental_design", "Experimental design"],
  ["variables", "Variables"],
  ["controlled_conditions", "Controlled conditions"],
  ["expected_outputs", "Expected outputs"],
  ["assumptions", "Assumptions"],
  ["risks_or_uncertainties", "Risks & uncertainties"],
  ["user_questions", "Open questions"],
];

const valueToMarkdown = (value: unknown): string => {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "string") return value.trim();
  if (Array.isArray(value)) {
    return value
      .map((v) =>
        v && typeof v === "object"
          ? `- ${Object.entries(v as Record<string, unknown>)
              .map(([k, val]) => `**${k}**: ${String(val)}`)
              .join("; ")}`
          : `- ${String(v)}`,
      )
      .filter(Boolean)
      .join("\n");
  }
  if (typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([k, val]) => `**${k}**: ${String(val)}`)
      .join("\n\n");
  }
  return String(value);
};

const SpecView = ({ report }: { report: Record<string, unknown> | null }): JSX.Element => {
  if (!report || Object.keys(report).length === 0) {
    return (
      <p className="text-sm italic text-muted-foreground">No experiment report was produced.</p>
    );
  }
  const known = new Set([...SPEC_FIELDS.map(([k]) => k), "title"]);
  const ordered = SPEC_FIELDS.map(
    ([key, label]) => [label, valueToMarkdown(report[key])] as const,
  ).filter(([, md]) => md);
  const extras = Object.entries(report)
    .filter(([k]) => !known.has(k))
    .map(([key, value]) => [key.replace(/_/g, " "), valueToMarkdown(value)] as const)
    .filter(([, md]) => md);
  return (
    <div className="space-y-4">
      {[...ordered, ...extras].map(([label, md]) => (
        <PanelSection key={label} title={label}>
          <MarkdownContent text={md} />
        </PanelSection>
      ))}
    </div>
  );
};

const PlanView = ({ plan }: { plan: PlanDetailResponse }): JSX.Element => (
  <div className="space-y-4">
    {plan.draft.trim() && (
      <PanelSection title="Original request">
        <div className="rounded-md border border-border/60 bg-muted/30 px-3 py-2">
          <MarkdownContent text={plan.draft.trim()} />
        </div>
      </PanelSection>
    )}
    <PanelSection title={`Workflow tasks (${plan.tasks.length})`}>
      {plan.tasks.length === 0 ? (
        <p className="text-sm italic text-muted-foreground">No workflow tasks were generated.</p>
      ) : (
        <ol className="space-y-1.5">
          {plan.tasks.map((task, idx) => (
            <li
              key={task.id}
              className="flex items-start gap-2.5 rounded-md border border-border/50 bg-card px-3 py-2"
            >
              <span className="mt-0.5 flex h-5 w-5 flex-none items-center justify-center rounded bg-muted text-[11px] font-medium tabular-nums text-muted-foreground">
                {idx + 1}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="truncate font-mono text-sm text-foreground">{task.id}</span>
                  {task.type && (
                    <Badge variant="secondary" className="h-4 px-1 font-mono text-[10px]">
                      {task.type}
                    </Badge>
                  )}
                </div>
                {task.source && (
                  <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                    {task.source}
                  </p>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}
    </PanelSection>
  </div>
);

const ScriptView = ({ source }: { source: string | null }): JSX.Element => {
  if (!source?.trim()) {
    return (
      <p className="text-sm italic text-muted-foreground">
        No runnable workflow source was generated.
      </p>
    );
  }
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[11px] text-muted-foreground">build_workflow.py</span>
        <CopyButton text={source} label="Copy source" />
      </div>
      <pre className="overflow-x-auto rounded-md border border-border/60 bg-muted/50 px-3 py-2.5 font-mono text-[11.5px] leading-relaxed text-foreground">
        {source}
      </pre>
    </div>
  );
};

const CodeBlock = ({
  text,
  filename,
  copyLabel,
  language,
}: {
  text: string;
  filename: string;
  copyLabel: string;
  language?: string;
}): JSX.Element => (
  <div className="space-y-2">
    <div className="flex items-center justify-between">
      <span className="font-mono text-[11px] text-muted-foreground">{filename}</span>
      <CopyButton text={text} label={copyLabel} />
    </div>
    <pre
      data-language={language}
      className="overflow-x-auto rounded-md border border-border/60 bg-muted/50 px-3 py-2.5 font-mono text-[11.5px] leading-relaxed text-foreground"
    >
      {text}
    </pre>
  </div>
);

const WorkflowIrView = ({ plan }: { plan: PlanDetailResponse }): JSX.Element => {
  // The curated workflow-spec YAML: inputs, tasks (purpose + typed I/O), edges.
  // Falls back to the task list when the IR artifact is absent (older plans).
  if (plan.workflowIrYaml?.trim())
    return (
      <CodeBlock
        text={plan.workflowIrYaml}
        filename="workflow_spec.yaml"
        copyLabel="Copy YAML"
        language="yaml"
      />
    );
  return <PlanView plan={plan} />;
};

// The Draft spec is the comprehensive specification rendered as ONE whole YAML:
// the experiment scheme + concretized parameters + resolved questions, WITH the
// workflow spec embedded as a `workflow_spec:` section (assembled server-side).
// Shown as a single un-split YAML block, not a sectioned document.
const SpecYamlView = ({ plan }: { plan: PlanDetailResponse }): JSX.Element => {
  if (!plan.experimentSpecYaml?.trim())
    return (
      <p className="text-sm italic text-muted-foreground">No concrete spec was drafted yet.</p>
    );
  return (
    <CodeBlock
      text={plan.experimentSpecYaml}
      filename="experiment_spec.yaml"
      copyLabel="Copy spec YAML"
      language="yaml"
    />
  );
};

const CapabilitiesView = ({ text }: { text: string | null }): JSX.Element => {
  if (!text?.trim())
    return <p className="text-sm italic text-muted-foreground">No capabilities were resolved.</p>;
  return <MarkdownContent text={text} />;
};

const InputSetView = ({ inputSet }: { inputSet: Record<string, unknown> | null }): JSX.Element => {
  if (!inputSet)
    return <p className="text-sm italic text-muted-foreground">No input set was generated.</p>;
  const axes = Array.isArray(inputSet.sweep_axes)
    ? (inputSet.sweep_axes as Record<string, unknown>[])
    : [];
  return (
    <div className="space-y-4">
      <PanelSection title="Parameter sweep">
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary" className="font-mono text-[11px]">
            {String(inputSet.strategy ?? "grid")}
          </Badge>
          <Badge variant="secondary" className="font-mono text-[11px]">
            {String(inputSet.total_runs ?? 1)} run{inputSet.total_runs === 1 ? "" : "s"}
          </Badge>
        </div>
      </PanelSection>
      <PanelSection title={`Axes (${axes.length})`}>
        {axes.length === 0 ? (
          <p className="text-sm italic text-muted-foreground">
            No swept axes — a single fixed-parameter run.
          </p>
        ) : (
          <ul className="space-y-1.5">
            {axes.map((axis, idx) => (
              <li
                key={String(axis.name ?? idx)}
                className="rounded-md border border-border/50 bg-card px-3 py-2"
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm text-foreground">{String(axis.name)}</span>
                  {axis.source != null && (
                    <Badge variant="secondary" className="h-4 px-1 font-mono text-[10px]">
                      {String(axis.source)}
                    </Badge>
                  )}
                </div>
                <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                  {Array.isArray(axis.values) ? axis.values.map(String).join(", ") : ""}
                </p>
              </li>
            ))}
          </ul>
        )}
      </PanelSection>
    </div>
  );
};

const DryRunView = ({ dryRun }: { dryRun: Record<string, unknown> | null }): JSX.Element => {
  if (!dryRun)
    return (
      <p className="text-sm italic text-muted-foreground">The workflow was not compiled yet.</p>
    );
  const meta = (dryRun.metadata ?? {}) as Record<string, unknown>;
  const ok = dryRun.status === "succeeded";
  return (
    <div className="space-y-4">
      <PanelSection title="Compile / dry-run">
        <div className="flex flex-wrap gap-2">
          <Badge variant={ok ? "secondary" : "destructive"} className="font-mono text-[11px]">
            {String(dryRun.status ?? "unknown")}
          </Badge>
          <Badge variant="secondary" className="font-mono text-[11px]">
            mode: {String(meta.mode ?? "run")}
          </Badge>
          <Badge variant="secondary" className="font-mono text-[11px]">
            exit {String(dryRun.exit_code ?? "?")}
          </Badge>
        </div>
      </PanelSection>
      <p className="text-xs text-muted-foreground">
        {ok
          ? "The generated source compiled and the workflow DAG built with the input-set parameters. No task bodies were executed — no real compute ran."
          : "Compilation failed; see the run's stderr artifact for details."}
      </p>
    </div>
  );
};

const ExecutionReportView = ({
  report,
}: {
  report: Record<string, unknown> | null;
}): JSX.Element => {
  if (!report)
    return (
      <p className="text-sm italic text-muted-foreground">No execution report was produced.</p>
    );
  const policy = (report.resource_policy ?? {}) as Record<string, unknown>;
  const env = (report.environment ?? {}) as Record<string, unknown>;
  const rows: [string, unknown][] = [
    ["Target", report.target_name],
    ["Scheduler", report.scheduler],
    ["Host", report.host],
    ["Account", report.account],
    ["Queue", report.queue],
    ["Partition", report.partition],
    ["Scratch root", report.scratch_root],
    ["Total runs", report.total_runs],
    ["Backend", policy.backend],
    ["Max runtime (s)", policy.max_runtime_s],
    ["Python", env.python_version],
  ];
  return (
    <div className="space-y-4">
      <PanelSection title="Where & how this will run">
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          {rows
            .filter(([, v]) => v != null && v !== "")
            .map(([label, value]) => (
              <div key={label} className="contents">
                <dt className="text-muted-foreground">{label}</dt>
                <dd className="font-mono text-foreground">{String(value)}</dd>
              </div>
            ))}
        </dl>
      </PanelSection>
      {Array.isArray(report.notes) && report.notes.length > 0 && (
        <PanelSection title="Notes">
          <ul className="list-disc space-y-0.5 pl-4 text-sm text-muted-foreground">
            {(report.notes as unknown[]).map((n) => (
              <li key={String(n)}>{String(n)}</li>
            ))}
          </ul>
        </PanelSection>
      )}
      <p className="text-xs text-muted-foreground/70">
        Descriptive only — molexp never submits a job from this report.
      </p>
    </div>
  );
};

const MultiFileView = ({ plan }: { plan: PlanDetailResponse }): JSX.Element => {
  // One file per task (workflow/<task>.py) + assembly, plus one test per task
  // (tests/test_<task>.py). A path selector keeps complex/many-task plans
  // readable — pick a file, see just that file. Single-file plans fall back to
  // the one source.
  const files = [...(plan.workflowFiles ?? []), ...(plan.testFiles ?? [])];
  const [active, setActive] = useState<string>("");
  if (files.length === 0) return <ScriptView source={plan.workflowSource} />;
  const current = files.find((f) => f.path === active) ?? files[0];
  return (
    <div className="space-y-3">
      <label className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="font-semibold uppercase tracking-wide">File</span>
        <select
          value={current.path}
          onChange={(e) => setActive(e.target.value)}
          className="min-w-0 flex-1 rounded-md border border-border/60 bg-card px-2 py-1.5 font-mono text-[11px] text-foreground"
        >
          {files.map((f) => (
            <option key={f.path} value={f.path}>
              {f.path}
            </option>
          ))}
        </select>
      </label>
      <CodeBlock
        text={current.source}
        filename={current.path}
        copyLabel="Copy file"
        language={current.path.endsWith(".py") ? "python" : undefined}
      />
    </div>
  );
};

const ReviewView = ({ plan }: { plan: PlanDetailResponse }): JSX.Element => {
  const review = plan.planReview;
  const findings = review && Array.isArray(review.findings) ? review.findings : [];
  const passed = review?.passed === true;
  return (
    <div className="space-y-4">
      <PanelSection title="Plan review">
        {review ? (
          <div className="space-y-2">
            <Badge variant={passed ? "secondary" : "destructive"} className="text-[11px]">
              {passed ? "passed — workflow is faithful to the spec" : "issues found"}
            </Badge>
            {typeof review.summary === "string" && review.summary && (
              <MarkdownContent text={review.summary} />
            )}
          </div>
        ) : (
          <p className="text-sm italic text-muted-foreground">
            No automated review was recorded for this plan.
          </p>
        )}
      </PanelSection>
      {findings.length > 0 && (
        <PanelSection title={`Findings (${findings.length})`}>
          <ul className="space-y-1.5">
            {findings.map((f, i) => {
              const rec = f as Record<string, unknown>;
              return (
                <li
                  key={`${String(rec.requirement ?? i)}`}
                  className="rounded-md border border-border/50 bg-card px-3 py-2 text-sm"
                >
                  <span className="font-medium">{String(rec.requirement ?? "requirement")}</span>
                  {rec.deviation != null && (
                    <p className="mt-0.5 text-muted-foreground">{String(rec.deviation)}</p>
                  )}
                </li>
              );
            })}
          </ul>
        </PanelSection>
      )}
      <p className="text-xs text-muted-foreground/70">
        This gate approves the verified plan before the execution report. (The CLI prompts here; the
        server path auto-approves.)
      </p>
    </div>
  );
};

const EmptyStage = ({ label }: { label: string }): JSX.Element => (
  <div className="flex flex-1 flex-col items-center justify-center gap-1.5 px-6 text-center">
    <FileQuestion className="h-6 w-6 text-muted-foreground/40" />
    <p className="text-sm text-muted-foreground">{label}</p>
    <p className="text-xs text-muted-foreground/70">This step produced no standalone document.</p>
  </div>
);

const PlanDeliverables = ({
  planRef,
  activeStageKind,
}: {
  planRef: PlanRef;
  activeStageKind: string;
}): JSX.Element => {
  const [plan, setPlan] = useState<PlanDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    workspaceApi
      .getPlan(planRef.projectId, planRef.experimentId, planRef.runId)
      .then((detail) => {
        if (!cancelled) setPlan(detail);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [planRef.projectId, planRef.experimentId, planRef.runId]);

  const title = plan?.title || planRef.title || "Experiment plan";
  const status = (plan?.status ?? "succeeded") as SemanticStatus;
  const stage = planStage(activeStageKind);

  const body = ((): JSX.Element => {
    if (loading)
      return (
        <div className="flex flex-1 items-center justify-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin text-info" />
          Loading deliverables…
        </div>
      );
    if (error || !plan)
      return (
        <div className="flex flex-1 items-center justify-center px-6 text-center text-sm text-muted-foreground">
          {error ?? "Plan deliverables are unavailable."}
        </div>
      );
    // The selected stage decides what document shows; a stage with no `view`
    // has no standalone document, so the panel is intentionally left empty.
    const inScroll = (node: JSX.Element): JSX.Element => (
      <ScrollArea className="min-h-0 flex-1">
        <div className="px-4 py-4">{node}</div>
      </ScrollArea>
    );
    switch (stage?.view) {
      case "report":
        return inScroll(<SpecView report={plan.experimentReport} />);
      case "spec":
        return inScroll(<SpecYamlView plan={plan} />);
      case "capabilities":
        return inScroll(<CapabilitiesView text={plan.capabilities} />);
      case "topology":
        return inScroll(<WorkflowIrView plan={plan} />);
      case "script":
        return inScroll(<MultiFileView plan={plan} />);
      case "inputs":
        return inScroll(<InputSetView inputSet={plan.inputSet} />);
      case "dryrun":
        return inScroll(<DryRunView dryRun={plan.dryRun} />);
      case "review":
        return inScroll(<ReviewView plan={plan} />);
      case "execution":
        return inScroll(<ExecutionReportView report={plan.executionReport} />);
      default:
        return <EmptyStage label={stage?.label ?? "No deliverable"} />;
    }
  })();

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-start gap-2 border-b border-border/60 bg-muted/20 px-4 py-2.5">
        <Package className="mt-0.5 h-4 w-4 flex-none text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-foreground" title={title}>
            {title}
          </p>
          <p className="font-mono text-[10px] text-muted-foreground">
            run {planRef.runId}
            {stage && <span className="ml-1.5 text-muted-foreground/70">· {stage.label}</span>}
          </p>
        </div>
        <StatusBadge status={status} size="sm" />
      </div>
      {body}
    </div>
  );
};

const ChatArtifacts = ({ artifacts }: { artifacts: Record<string, unknown>[] }): JSX.Element => (
  <div className="flex h-full flex-col">
    <div className="flex items-center gap-2 border-b border-border/60 bg-muted/20 px-4 py-2.5">
      <Package className="h-4 w-4 flex-none text-muted-foreground" />
      <p className="text-sm font-semibold text-foreground">
        Artifacts <span className="font-normal text-muted-foreground">· {artifacts.length}</span>
      </p>
    </div>
    <ScrollArea className="min-h-0 flex-1">
      <div className="space-y-3 px-4 py-4">
        {artifacts.map((artifact, idx) => {
          const title = typeof artifact.title === "string" && artifact.title ? artifact.title : "";
          const key = `${String(artifact.kind ?? "?")}:${title || idx}`;
          return <ArtifactBody key={key} payload={artifact} />;
        })}
      </div>
    </ScrollArea>
  </div>
);

/**
 * The right-hand deliverables panel. Renders a PlanMode plan (Spec/Plan/Script)
 * when the session carries a plan locator, otherwise a chat session's inline
 * artifacts. The parent only mounts this when {@link hasDeliverables} is true,
 * so the empty branch is a defensive fallback.
 */
export const DeliverablesPanel = ({
  events,
  activeStageKind,
}: {
  events: ApiSessionEvent[];
  activeStageKind: string;
}): JSX.Element => {
  const planRef = useMemo(() => derivePlanRef(events), [events]);
  const artifacts = useMemo(() => collectArtifacts(events), [events]);

  if (planRef) return <PlanDeliverables planRef={planRef} activeStageKind={activeStageKind} />;
  if (artifacts.length > 0) return <ChatArtifacts artifacts={artifacts} />;
  return (
    <div className="flex h-full items-center justify-center px-6 text-center text-sm text-muted-foreground">
      Deliverables will appear here as the agent produces them.
    </div>
  );
};
