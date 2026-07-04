import { Check, ClipboardCopy, FileQuestion, Loader2, Package, WrapText } from "lucide-react";
import { type JSX, useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { PlanDetailResponse } from "@/api/generated/models/PlanDetailResponse";
import { StatusBadge } from "@/app/components/entity";
import type { PlanRef } from "@/app/renderers/agentEvents";
import { collectArtifacts, derivePlanRef } from "@/app/renderers/agentEvents";
import { runPath } from "@/app/entities/paths";
import { workspaceApi } from "@/app/state/api";
import type { ApiSessionEvent, SemanticStatus } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { MarkdownContent } from "@/components/ui/markdown";
import { ScrollArea } from "@/components/ui/scroll-area";
import { highlightCode, type TokenKind } from "@/lib/highlight";
import { cn } from "@/lib/utils";
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
    <CodeBlock
      text={source}
      filename="build_workflow.py"
      copyLabel="Copy source"
      language="python"
    />
  );
};

// Semantic-token colors for the read-only code panels — theme tokens only,
// so they track light/dark like every other surface.
const TOKEN_CLASS: Record<Exclude<TokenKind, "plain">, string> = {
  comment: "italic text-muted-foreground",
  string: "text-success",
  keyword: "font-medium text-info",
  number: "text-warning",
  decorator: "text-warning",
  key: "text-info",
};

const HighlightedCode = ({
  text,
  language,
}: {
  text: string;
  language?: string;
}): JSX.Element => {
  const tokens = useMemo(() => highlightCode(text, language), [text, language]);
  let offset = 0;
  return (
    <code>
      {tokens.map((token) => {
        const key = offset;
        offset += token.text.length;
        return (
          <span key={key} className={token.kind === "plain" ? undefined : TOKEN_CLASS[token.kind]}>
            {token.text}
          </span>
        );
      })}
    </code>
  );
};

// Reviewers read whole YAML specs and python sources in this panel, often in a
// narrow split — so long lines soft-wrap by default (never a 60-char guillotine)
// with an explicit toggle to the classic one-line-per-row + horizontal scroll.
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
}): JSX.Element => {
  const [wrap, setWrap] = useState(true);
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="min-w-0 truncate font-mono text-[11px] text-muted-foreground">
          {filename}
        </span>
        <div className="flex flex-none items-center gap-1.5">
          <button
            type="button"
            onClick={() => setWrap((prev) => !prev)}
            aria-pressed={wrap}
            title={
              wrap
                ? "Soft-wrapping long lines — click to scroll them instead"
                : "Long lines scroll horizontally — click to soft-wrap"
            }
            className="inline-flex items-center gap-1 rounded-md border border-border/60 bg-card px-2 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <WrapText className="h-3 w-3" />
            {wrap ? "Wrap" : "No wrap"}
          </button>
          <CopyButton text={text} label={copyLabel} />
        </div>
      </div>
      <pre
        data-language={language}
        className={cn(
          "rounded-md border border-border/60 bg-muted/50 px-3 py-2.5 font-mono text-[11.5px] leading-relaxed text-foreground",
          wrap
            ? "whitespace-pre-wrap break-words"
            : "overflow-x-auto [scrollbar-width:thin] [&::-webkit-scrollbar]:h-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border [&::-webkit-scrollbar-track]:bg-transparent",
        )}
      >
        <HighlightedCode text={text} language={language} />
      </pre>
    </div>
  );
};

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

// One entry of `capabilitySelection.selected` — the server may record bare
// capability-id strings or objects carrying the id plus a per-pick rationale.
interface SelectedCapability {
  id: string;
  detail?: string;
}

const asSelectedCapability = (entry: unknown): SelectedCapability => {
  if (typeof entry === "string") return { id: entry };
  if (entry && typeof entry === "object") {
    const rec = entry as Record<string, unknown>;
    const id = rec.capability_id ?? rec.id ?? rec.name;
    const detail = rec.reason ?? rec.purpose ?? rec.notes ?? rec.description;
    return {
      id: typeof id === "string" && id ? id : JSON.stringify(entry),
      detail: typeof detail === "string" && detail ? detail : undefined,
    };
  }
  return { id: String(entry) };
};

// The Resolve-capabilities deliverable is the LLM's SELECTION — the minimal
// capability subset this experiment binds — not the grounded catalog prompt
// ("Do NOT invent capability_ids…" is agent instruction, not user content).
// The full catalog stays reachable, folded behind a <details>.
const CapabilitiesView = ({ plan }: { plan: PlanDetailResponse }): JSX.Element => {
  const catalog = plan.capabilities?.trim() ?? "";
  const selection = plan.capabilitySelection;

  if (!selection) {
    // Older plans recorded no selection artifact — show whatever was grounded.
    if (!catalog)
      return <p className="text-sm italic text-muted-foreground">No capabilities were resolved.</p>;
    return <MarkdownContent text={catalog} />;
  }

  const selected = Array.isArray(selection.selected)
    ? (selection.selected as unknown[]).map(asSelectedCapability)
    : [];
  const notes = typeof selection.notes === "string" ? selection.notes.trim() : "";

  return (
    <div className="space-y-4">
      <PanelSection title={`Selected capabilities (${selected.length})`}>
        {selected.length === 0 ? (
          <MarkdownContent
            text={
              notes
                ? `None — ${notes}`
                : "None — this experiment binds no capability from the catalog."
            }
          />
        ) : (
          <ul className="space-y-1.5">
            {selected.map((cap) => (
              <li key={cap.id} className="rounded-md border border-border/50 bg-card px-3 py-2">
                <span className="break-all font-mono text-sm text-foreground">{cap.id}</span>
                {cap.detail && <p className="mt-0.5 text-xs text-muted-foreground">{cap.detail}</p>}
              </li>
            ))}
          </ul>
        )}
      </PanelSection>
      {selected.length > 0 && notes && (
        <PanelSection title="Selection rationale">
          <MarkdownContent text={notes} />
        </PanelSection>
      )}
      {catalog && (
        <details className="rounded-md border border-border/60 bg-card">
          <summary className="cursor-pointer select-none px-3 py-2 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground">
            Full grounded catalog
          </summary>
          <div className="border-t border-border/60 px-3 py-2">
            <MarkdownContent text={catalog} />
          </div>
        </details>
      )}
    </div>
  );
};

const InputSetView = ({ inputSet }: { inputSet: Record<string, unknown> | null }): JSX.Element => {
  if (!inputSet)
    return <p className="text-sm italic text-muted-foreground">No input set was generated.</p>;
  const axes = Array.isArray(inputSet.sweep_axes)
    ? (inputSet.sweep_axes as Record<string, unknown>[])
    : [];
  const fixed =
    inputSet.fixed_params && typeof inputSet.fixed_params === "object"
      ? Object.entries(inputSet.fixed_params as Record<string, unknown>)
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
      {fixed.length > 0 && (
        <PanelSection title={`Fixed params (${fixed.length})`}>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            {fixed.map(([name, value]) => (
              <div key={name} className="contents">
                <dt className="font-mono text-muted-foreground">{name}</dt>
                <dd className="font-mono text-foreground">{JSON.stringify(value)}</dd>
              </div>
            ))}
          </dl>
          <p className="text-xs text-muted-foreground/70">
            Passed whole into every cell — list-valued inputs the workflow scans internally.
          </p>
        </PanelSection>
      )}
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

// Final-report prose fields, in reading order (mirrors the FinalReport schema).
const FINAL_REPORT_FIELDS: [string, string][] = [
  ["objective", "Objective"],
  ["methods_summary", "Methods"],
  ["test_summary", "Tests"],
  ["execution_summary", "Execution"],
  ["results", "Results"],
  ["conclusions", "Conclusions"],
  ["limitations", "Limitations"],
  ["next_steps", "Next steps"],
];

const FinalReportView = ({ plan }: { plan: PlanDetailResponse }): JSX.Element => {
  const report = plan.finalReport;
  const execution = plan.execution;
  if (!report)
    return (
      <p className="text-sm italic text-muted-foreground">
        The workflow has not been executed — run <code className="font-mono">molexp plan
        --execute</code> to produce the final report.
      </p>
    );
  const ok = execution?.status === "succeeded";
  return (
    <div className="space-y-4">
      {execution && (
        <PanelSection title="Real execution">
          <div className="flex flex-wrap gap-2">
            <Badge variant={ok ? "secondary" : "destructive"} className="font-mono text-[11px]">
              {String(execution.status ?? "unknown")}
            </Badge>
            <Badge variant="secondary" className="font-mono text-[11px]">
              exit {String(execution.exit_code ?? "?")}
            </Badge>
          </div>
        </PanelSection>
      )}
      {typeof report.title === "string" && report.title && (
        <p className="text-sm font-semibold text-foreground">{String(report.title)}</p>
      )}
      {FINAL_REPORT_FIELDS.map(([key, label]) => {
        const value = report[key];
        if (typeof value !== "string" || !value.trim()) return null;
        return (
          <PanelSection key={key} title={label}>
            <MarkdownContent text={value} />
          </PanelSection>
        );
      })}
      <p className="text-xs text-muted-foreground/70">
        Grounded in the run&apos;s persisted TestResult + ExecutionResult artifacts — every number
        comes from a real output.
      </p>
    </div>
  );
};

const AuditReportView = ({ report }: { report: Record<string, unknown> | null }): JSX.Element => {
  if (!report)
    return (
      <p className="text-sm italic text-muted-foreground">
        No audit report — it is generated at the end of a real execution.
      </p>
    );
  const entries = Object.entries(report).filter(([, v]) => v == null || typeof v !== "object");
  return (
    <div className="space-y-4">
      <PanelSection title="Audit trail">
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          {entries.map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="text-muted-foreground">{key}</dt>
              <dd className="font-mono text-foreground">{String(value ?? "—")}</dd>
            </div>
          ))}
        </dl>
      </PanelSection>
      <PanelSection title="Full record">
        <CodeBlock
          text={JSON.stringify(report, null, 2)}
          filename="audit_report.json"
          copyLabel="Copy JSON"
        />
      </PanelSection>
      <p className="text-xs text-muted-foreground/70">
        Every stage, artifact, and lineage edge of this run is also queryable in{" "}
        <code className="font-mono">harness.sqlite</code>.
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
        language={
          current.path.endsWith(".py")
            ? "python"
            : /\.ya?ml$/.test(current.path)
              ? "yaml"
              : undefined
        }
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
  const navigate = useNavigate();
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
        return inScroll(<CapabilitiesView plan={plan} />);
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
      case "final":
        return inScroll(<FinalReportView plan={plan} />);
      case "audit":
        return inScroll(<AuditReportView report={plan.auditReport} />);
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
            <button
              type="button"
              className="underline decoration-dotted underline-offset-2 hover:text-foreground"
              title="Open this run in the Projects tree"
              onClick={() => navigate(runPath(planRef.projectId, planRef.experimentId, planRef.runId))}
            >
              run {planRef.runId}
            </button>
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
