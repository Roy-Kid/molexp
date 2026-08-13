/**
 * Inline experiment plan book (12-section markdown document).
 *
 * Loads `GET /plans/{runId}` so the plan remains readable after page refresh.
 * Prefers plan_report / experiment_report `body_md`, else projects the
 * experiment_plan board into the canonical outline.
 */

import { ClipboardList } from "lucide-react";
import { type JSX, useEffect, useMemo, useState } from "react";
import type { PlanDetailResponse } from "@/api/generated/models/PlanDetailResponse";
import { workspaceApi } from "@/app/state/api";
import { MarkdownContent } from "@/components/ui/markdown";
import { ProgressSpinner } from "@/components/ui/progress-spinner";
import { cn } from "@/lib/utils";
import { renderExperimentPlanDocument } from "./experimentPlanDocument";

const reportBodyMd = (report: Record<string, unknown> | null | undefined): string | null => {
  if (!report || typeof report !== "object") return null;
  if (typeof report.body_md === "string" && report.body_md.trim()) return report.body_md.trim();
  if (typeof report.markdown === "string" && report.markdown.trim()) return report.markdown.trim();
  if (typeof report.summary_md === "string" && report.summary_md.trim()) {
    return report.summary_md.trim();
  }
  // Reconstruct a thin document from legacy structured ExperimentReport fields.
  const title = typeof report.title === "string" ? report.title : "";
  const objective = typeof report.objective === "string" ? report.objective : "";
  const system = typeof report.system_description === "string" ? report.system_description : "";
  const design = typeof report.experimental_design === "string" ? report.experimental_design : "";
  if (!title && !objective && !system && !design) return null;
  return renderExperimentPlanDocument({
    title: title || "Experiment Plan",
    objective,
    system,
    design,
    background: typeof report.background === "string" ? report.background : "",
    hypotheses:
      typeof report.scientific_hypothesis === "string" && report.scientific_hypothesis
        ? [report.scientific_hypothesis]
        : [],
    variables: Array.isArray(report.variables) ? report.variables.map(String) : [],
    controlled: Array.isArray(report.controlled_conditions)
      ? report.controlled_conditions.map(String)
      : [],
    expected: Array.isArray(report.expected_outputs) ? report.expected_outputs.map(String) : [],
    risks: Array.isArray(report.risks_or_uncertainties)
      ? report.risks_or_uncertainties.map(String)
      : [],
    tasks: [],
  });
};

export const PlanDocumentCard = ({
  projectId,
  experimentId,
  runId,
  className,
  compact = false,
}: {
  projectId: string;
  experimentId: string;
  runId: string;
  className?: string;
  compact?: boolean;
}): JSX.Element | null => {
  const [plan, setPlan] = useState<PlanDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!projectId || !experimentId || !runId) {
      setLoading(false);
      setError("Missing plan locator (project / experiment / run).");
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    workspaceApi
      .getPlan(projectId, experimentId, runId)
      .then((detail) => {
        if (!cancelled) setPlan(detail);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, experimentId, runId]);

  const documentMd = useMemo(() => {
    if (!plan) return "";
    const fromReport =
      reportBodyMd(plan.planReport as Record<string, unknown> | null) ||
      reportBodyMd(plan.experimentReport as Record<string, unknown> | null);
    if (fromReport) return fromReport;

    const ep = plan.experimentPlan as
      | { spec?: Record<string, unknown>; board?: { tasks?: unknown[] } }
      | null
      | undefined;
    const frozen = plan.frozenExperimentPlan as
      | { spec?: Record<string, unknown>; board?: { tasks?: unknown[] } }
      | null
      | undefined;
    const source = ep?.board ? ep : frozen;
    const spec = (source?.spec ?? {}) as Record<string, unknown>;
    const rawTasks = source?.board?.tasks;
    const tasks = Array.isArray(rawTasks)
      ? rawTasks
          .filter((t): t is Record<string, unknown> => Boolean(t) && typeof t === "object")
          .map((t) => {
            const acceptanceRaw = t.acceptance;
            const acceptance = Array.isArray(acceptanceRaw)
              ? acceptanceRaw.map(String)
              : typeof acceptanceRaw === "string" && acceptanceRaw
                ? [acceptanceRaw]
                : [];
            return {
              id: String(t.id ?? ""),
              name: String(t.name ?? t.id ?? "task"),
              status: String(t.status ?? "pending"),
              acceptance,
            };
          })
          .filter((t) => t.id)
      : (plan.tasks ?? []).map((t) => ({
          id: t.id,
          name: t.source || t.id,
          status: t.type || "pending",
          acceptance: [] as string[],
        }));

    return renderExperimentPlanDocument({
      title: plan.title || String(spec.title ?? "Experiment Plan"),
      objective: String(spec.objective ?? plan.draft ?? ""),
      system: String(spec.system_description ?? spec.system ?? ""),
      design: String(spec.experimental_design ?? spec.design ?? ""),
      background: String(spec.background ?? ""),
      gap: String(spec.knowledge_gap ?? ""),
      motivation: String(spec.motivation ?? ""),
      questions: Array.isArray(spec.scientific_questions)
        ? spec.scientific_questions.map(String)
        : Array.isArray(spec.questions)
          ? spec.questions.map(String)
          : [],
      hypotheses: Array.isArray(spec.hypotheses)
        ? spec.hypotheses.map(String)
        : typeof spec.scientific_hypothesis === "string" && spec.scientific_hypothesis
          ? [spec.scientific_hypothesis]
          : [],
      variables: Array.isArray(spec.variables) ? spec.variables.map(String) : [],
      dependents: Array.isArray(spec.dependent_variables)
        ? spec.dependent_variables.map(String)
        : [],
      controlled: Array.isArray(spec.controlled_conditions)
        ? spec.controlled_conditions.map(String)
        : Array.isArray(spec.controlled_variables)
          ? spec.controlled_variables.map(String)
          : [],
      methods: Array.isArray(spec.computational_methods)
        ? spec.computational_methods.map(String)
        : [],
      risks: Array.isArray(spec.risks_or_uncertainties)
        ? spec.risks_or_uncertainties.map(String)
        : Array.isArray(spec.risks)
          ? spec.risks.map(String)
          : [],
      expected: Array.isArray(spec.expected_outputs) ? spec.expected_outputs.map(String) : [],
      success: Array.isArray(spec.success_criteria) ? spec.success_criteria.map(String) : [],
      tasks,
    });
  }, [plan]);

  if (loading) {
    return (
      <div
        className={cn(
          "flex items-center gap-2 rounded-control border border-border/60 bg-muted/30 px-3 py-2 text-label text-muted-foreground",
          className,
        )}
      >
        <ProgressSpinner className="text-info" label="Loading experiment plan" />
        Loading experiment plan…
      </div>
    );
  }

  if (error || !plan) {
    return (
      <div
        className={cn(
          "rounded-control border border-border/60 bg-muted/20 px-3 py-2 text-label text-muted-foreground",
          className,
        )}
      >
        {error ?? "Experiment plan is unavailable."}
      </div>
    );
  }

  const title = plan.title || "Experiment Plan";

  return (
    <div
      className={cn(
        "space-y-2 rounded-control border border-info/25 bg-info-soft/15 px-3 py-3",
        className,
      )}
    >
      <div className="flex items-start gap-2">
        <ClipboardList className="mt-0.5 h-4 w-4 flex-none text-info" />
        <div className="min-w-0 flex-1">
          <p className="text-body-lg font-semibold text-foreground">{title}</p>
          <p className="font-mono text-micro text-muted-foreground">
            {projectId}/{experimentId} · run {runId}
          </p>
        </div>
      </div>
      <div
        className={cn(
          "border-t border-border/50 pt-2 text-label leading-relaxed text-foreground",
          compact ? "max-h-72 overflow-auto" : "max-h-128 overflow-auto",
        )}
      >
        <MarkdownContent text={documentMd || "_(empty plan document)_"} />
      </div>
    </div>
  );
};
