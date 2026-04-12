import {
  Activity,
  AlertCircle,
  ArrowLeft,
  Box,
  CheckCircle2,
  Circle,
  Clock,
  FileText,
  Loader2,
  Terminal,
  XCircle,
} from "lucide-react";
import { type ComponentType, useEffect, useMemo, useState } from "react";
import { buildMetadataFields } from "@/app/renderers/metadata";
import { useUrlState } from "@/app/state/useUrlState";
import type { RendererProps, SemanticStatus } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// ── Types ────────────────────────────────────────────────────────────────────

interface RunDetail {
  slurmJobId?: string | null;
  molqJobId?: string | null;
  error?: { type: string; message: string } | null;
}

interface RunLogs {
  stdout?: string | null;
  stderr?: string | null;
}

interface WorkflowStep {
  index: number;
  status: string;
  step_outputs: Record<string, unknown>;
}

interface RunExecution {
  execution_id?: string | null;
  status: string;
  steps: WorkflowStep[];
  end?: Record<string, unknown> | null;
}

// ── Sub-components ───────────────────────────────────────────────────────────

const MetricStat = ({
  label,
  value,
  icon: Icon,
  colorClass,
}: {
  label: string;
  value: number | string;
  icon: ComponentType<{ className?: string }>;
  colorClass?: string;
}) => (
  <div className="flex flex-col gap-1">
    <div className="flex items-center gap-2 text-muted-foreground mb-1">
      <Icon className={`h-4 w-4 ${colorClass || "text-muted-foreground"}`} />
      <span className="text-xs font-medium uppercase tracking-wider">{label}</span>
    </div>
    <span className="text-3xl font-light tracking-tight text-foreground">{value}</span>
  </div>
);

const StatusBadge = ({ status }: { status: SemanticStatus }) => {
  if (status === "succeeded") {
    return (
      <Badge
        variant="default"
        className="bg-green-600 hover:bg-green-700 gap-1.5 pl-1.5 pr-2.5 py-1"
      >
        <CheckCircle2 className="h-4 w-4" /> Succeeded
      </Badge>
    );
  }
  if (status === "failed") {
    return (
      <Badge variant="destructive" className="gap-1.5 pl-1.5 pr-2.5 py-1">
        <XCircle className="h-4 w-4" /> Failed
      </Badge>
    );
  }
  if (status === "running") {
    return (
      <Badge
        variant="secondary"
        className="bg-blue-100 text-blue-800 hover:bg-blue-200 gap-1.5 pl-1.5 pr-2.5 py-1 animate-pulse"
      >
        <Activity className="h-4 w-4" /> Running
      </Badge>
    );
  }
  if (status === "cancelled") {
    return (
      <Badge variant="secondary" className="gap-1.5 pl-1.5 pr-2.5 py-1">
        <AlertCircle className="h-4 w-4" /> Cancelled
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="gap-1.5 pl-1.5 pr-2.5 py-1">
      <Clock className="h-4 w-4" /> {status}
    </Badge>
  );
};

const StepStatusIcon = ({ status }: { status: string }) => {
  if (status === "success")
    return <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />;
  if (status === "error")
    return <XCircle className="h-4 w-4 text-red-500 shrink-0" />;
  if (status === "running")
    return <Loader2 className="h-4 w-4 text-blue-500 animate-spin shrink-0" />;
  return <Circle className="h-4 w-4 text-muted-foreground shrink-0" />;
};

// ── Logs tab ─────────────────────────────────────────────────────────────────

const LogsPanel = ({
  projectId,
  experimentId,
  runId,
}: {
  projectId: string;
  experimentId: string;
  runId: string;
}) => {
  const [logs, setLogs] = useState<RunLogs | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeLog, setActiveLog] = useState<"stdout" | "stderr">("stdout");

  useEffect(() => {
    setLoading(true);
    fetch(`/api/projects/${projectId}/experiments/${experimentId}/runs/${runId}/logs`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => setLogs(data))
      .finally(() => setLoading(false));
  }, [projectId, experimentId, runId]);

  const content = activeLog === "stdout" ? logs?.stdout : logs?.stderr;

  return (
    <div className="flex h-full flex-col bg-slate-950 text-slate-50">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-800 bg-slate-900 text-xs font-mono text-slate-400 shrink-0">
        <Terminal className="h-3 w-3" />
        <button
          onClick={() => setActiveLog("stdout")}
          className={`px-2 py-0.5 rounded ${activeLog === "stdout" ? "bg-slate-700 text-slate-100" : "hover:bg-slate-800"}`}
        >
          job.out
        </button>
        <button
          onClick={() => setActiveLog("stderr")}
          className={`px-2 py-0.5 rounded ${activeLog === "stderr" ? "bg-slate-700 text-slate-100" : "hover:bg-slate-800"}`}
        >
          job.err
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 p-4 font-mono text-xs overflow-auto whitespace-pre-wrap leading-relaxed">
        {loading ? (
          <span className="opacity-50 italic">Loading…</span>
        ) : content ? (
          content
        ) : (
          <span className="opacity-50 italic">
            {logs ? "No content (file is empty or not written yet)." : "Log files not available."}
          </span>
        )}
      </div>
    </div>
  );
};

// ── Execution tab ─────────────────────────────────────────────────────────────

const ExecutionPanel = ({
  projectId,
  experimentId,
  runId,
}: {
  projectId: string;
  experimentId: string;
  runId: string;
}) => {
  const [exec, setExec] = useState<RunExecution | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/projects/${projectId}/experiments/${experimentId}/runs/${runId}/execution`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setExec)
      .finally(() => setLoading(false));
  }, [projectId, experimentId, runId]);

  if (loading)
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
        <Loader2 className="h-4 w-4 animate-spin mr-2" /> Loading execution state…
      </div>
    );

  if (!exec || exec.status === "not_started")
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground">
        <Circle className="h-10 w-10 mb-3 opacity-20" />
        <div>No execution recorded yet.</div>
      </div>
    );

  return (
    <div className="flex-1 overflow-auto p-6 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3 pb-2 border-b">
        <span className="text-xs text-muted-foreground font-mono">{exec.execution_id}</span>
        <Badge
          variant="outline"
          className={
            exec.status === "completed"
              ? "border-green-500 text-green-600"
              : exec.status === "running"
                ? "border-blue-500 text-blue-600 animate-pulse"
                : "border-red-500 text-red-600"
          }
        >
          {exec.status}
        </Badge>
      </div>

      {/* Steps */}
      <div className="space-y-2">
        {exec.steps.length === 0 ? (
          <p className="text-sm text-muted-foreground italic">No steps recorded.</p>
        ) : (
          exec.steps.map((step) => {
            const outputKeys = Object.keys(step.step_outputs);
            return (
              <div
                key={step.index}
                className="flex items-start gap-3 p-3 rounded-lg border bg-muted/20"
              >
                <StepStatusIcon status={step.status} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Step {step.index}</span>
                    <span className="text-xs text-muted-foreground capitalize">{step.status}</span>
                  </div>
                  {outputKeys.length > 0 && (
                    <div className="mt-1 text-xs text-muted-foreground font-mono">
                      outputs: {outputKeys.join(", ")}
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* End outputs */}
      {exec.end && (
        <div className="mt-4 p-4 rounded-lg border bg-green-500/5 border-green-500/20">
          <div className="text-xs font-medium text-green-700 mb-2 uppercase tracking-wider">
            Final outputs
          </div>
          <div className="text-xs font-mono space-y-1">
            {Object.entries(
              (exec.end as { step_outputs?: Record<string, unknown> }).step_outputs ?? {},
            ).map(([k, v]) => (
              <div key={k}>
                <span className="text-muted-foreground">{k}:</span>{" "}
                <span>{typeof v === "object" ? JSON.stringify(v) : String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ── Main component ───────────────────────────────────────────────────────────

export const RunViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const fields = buildMetadataFields(selection, snapshot);
  const { setSelection } = useUrlState();

  const run = useMemo(() => {
    return snapshot.runs.find((r) => r.id === selection.objectId);
  }, [snapshot.runs, selection.objectId]);

  const [detail, setDetail] = useState<RunDetail | null>(null);

  // Fetch full run details (slurmJobId, molqJobId, error) on mount
  useEffect(() => {
    if (!run) return;
    fetch(
      `/api/projects/${run.projectId}/experiments/${run.experimentId}/runs/${run.id}`,
    )
      .then((r) => (r.ok ? r.json() : null))
      .then(setDetail);
  }, [run?.id]);

  if (!run) {
    return <div className="p-8 text-muted-foreground">Run not found.</div>;
  }

  const displayFields = fields.filter(
    (f) => !["Run", "Status", "Summary", "Project", "Experiment"].includes(f.label),
  );

  return (
    <div className="flex h-full flex-col bg-background">
      {/* Header */}
      <div className="flex flex-col gap-4 px-8 py-6 border-b bg-background">
        {/* Back button */}
        <button
          onClick={() =>
            setSelection({ objectType: "experiment", objectId: run.experimentId })
          }
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors w-fit"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to experiment
        </button>

        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <FileText className="h-6 w-6 text-blue-600" />
              </div>
              <h1 className="text-2xl font-bold tracking-tight text-foreground">{run.name}</h1>
              <Badge variant="outline" className="font-mono text-xs text-muted-foreground">
                {run.id}
              </Badge>
            </div>

            <div className="flex items-center gap-2 pl-[3.25rem] flex-wrap">
              <StatusBadge status={run.status} />

              {/* SLURM / molq job IDs */}
              {detail?.slurmJobId && (
                <Badge variant="outline" className="font-mono text-xs gap-1">
                  <Terminal className="h-3 w-3" />
                  SLURM {detail.slurmJobId}
                </Badge>
              )}
              {detail?.molqJobId && (
                <Badge variant="outline" className="font-mono text-xs text-muted-foreground">
                  molq {detail.molqJobId.slice(0, 8)}
                </Badge>
              )}
            </div>

            {/* Error banner */}
            {detail?.error && (
              <div className="mt-2 pl-[3.25rem] flex items-start gap-2 text-sm text-red-600">
                <XCircle className="h-4 w-4 mt-0.5 shrink-0" />
                <span>
                  <span className="font-medium">{detail.error.type}:</span>{" "}
                  {detail.error.message}
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-10 pl-[3.25rem]">
          <MetricStat
            label="Updated"
            value={new Date(run.updatedAt).toLocaleTimeString()}
            icon={Clock}
          />
        </div>
      </div>

      {/* Tabs */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <Tabs defaultValue="overview" className="flex-1 flex flex-col">
          <div className="px-8 py-3 border-b bg-background">
            <TabsList className="h-auto w-fit justify-start bg-muted/20 p-1 rounded-lg">
              <TabsTrigger value="overview" className="px-4 py-2 rounded-md font-medium text-sm">
                Overview
              </TabsTrigger>
              <TabsTrigger value="logs" className="px-4 py-2 rounded-md font-medium text-sm">
                Logs
              </TabsTrigger>
              <TabsTrigger value="artifacts" className="px-4 py-2 rounded-md font-medium text-sm">
                Artifacts
              </TabsTrigger>
              <TabsTrigger value="execution" className="px-4 py-2 rounded-md font-medium text-sm">
                Execution
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Overview */}
          <TabsContent value="overview" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
            <div className="flex-1 overflow-auto">
              <table className="w-full text-sm text-left">
                <tbody className="divide-y divide-border/50">
                  {displayFields.length === 0 ? (
                    <tr>
                      <td colSpan={2} className="py-12 text-center text-muted-foreground">
                        No additional details.
                      </td>
                    </tr>
                  ) : (
                    displayFields.map((field) => (
                      <tr
                        key={`field-${field.label}`}
                        className="hover:bg-muted/30 transition-colors"
                      >
                        <td className="py-3 px-6 font-medium text-muted-foreground w-[200px]">
                          {field.label}
                        </td>
                        <td className="py-3 px-6 font-mono text-xs">{field.value}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </TabsContent>

          {/* Logs */}
          <TabsContent value="logs" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
            <LogsPanel
              projectId={run.projectId}
              experimentId={run.experimentId}
              runId={run.id}
            />
          </TabsContent>

          {/* Artifacts */}
          <TabsContent value="artifacts" className="flex-1 p-0 flex flex-col">
            <div className="flex-1 p-8 flex flex-col items-center justify-center text-muted-foreground">
              <Box className="h-12 w-12 mb-4 opacity-20" />
              <div>No artifacts generated.</div>
            </div>
          </TabsContent>

          {/* Execution */}
          <TabsContent value="execution" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
            <ExecutionPanel
              projectId={run.projectId}
              experimentId={run.experimentId}
              runId={run.id}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
