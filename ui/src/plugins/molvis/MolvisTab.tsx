import { Atom, FileText } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { EmptyState, OverviewSection } from "@/app/components/entity";
import type { LammpsLogResponse, LammpsThermoStage } from "@/app/state/api";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import { Plot } from "@/lib/plot";
import type { DiscoveredFile } from "@/plugins/types";
import { TrajectoryViewer } from "./TrajectoryViewer";

type MolvisTabProps = RendererProps & { discoveredFiles?: DiscoveredFile[] };

const PALETTE = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#db2777",
  "#65a30d",
];

const TRAJECTORY_PATTERNS = /\.(lammpstrj|lmptrj|lammpsdump|dump|xyz|extxyz|pdb)$/i;
const LOG_PATTERNS = /(^log\.lammps$|\.lammps\.log$|^lmp\.log$)/i;

const isLogFile = (file: DiscoveredFile): boolean => LOG_PATTERNS.test(file.name);

const formatBytes = (size?: number | null): string => {
  if (!Number.isFinite(size ?? Number.NaN)) {
    return "—";
  }
  const value = Number(size);
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
};

interface ThermoChartProps {
  stage: LammpsThermoStage;
  columnIndex: number;
  color: string;
}

const ThermoChart = ({ stage, columnIndex, color }: ThermoChartProps): JSX.Element => {
  const columns = stage.columns ?? [];
  const rows = stage.rows ?? [];
  const stepIndex = columns.indexOf("Step");
  const xs = rows.map((row, idx) => (stepIndex >= 0 ? row[stepIndex] : idx));
  const ys = rows.map((row) => row[columnIndex]);

  const data: Record<string, unknown>[] = [
    {
      type: "scatter",
      mode: "lines",
      name: columns[columnIndex],
      x: xs,
      y: ys,
      line: { color, width: 2 },
      hovertemplate: "%{y:.6g}<extra></extra>",
    },
  ];

  const layout: Record<string, unknown> = {
    autosize: true,
    margin: { l: 48, r: 16, t: 8, b: 36 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "ui-sans-serif, system-ui, sans-serif", size: 11, color: "#64748b" },
    xaxis: {
      title: { text: "Step", font: { size: 10 } },
      gridcolor: "rgba(148,163,184,0.18)",
      zerolinecolor: "rgba(148,163,184,0.3)",
      tickfont: { size: 10 },
    },
    yaxis: {
      gridcolor: "rgba(148,163,184,0.18)",
      zerolinecolor: "rgba(148,163,184,0.3)",
      tickfont: { size: 10 },
    },
    hovermode: "x unified",
    showlegend: false,
  };

  const config: Record<string, unknown> = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "toggleSpikelines"],
    displayModeBar: "hover",
  };

  return (
    <Plot
      data={data}
      layout={layout}
      config={config}
      useResizeHandler
      style={{ width: "100%", height: "220px" }}
    />
  );
};

interface ThermoStageProps {
  stage: LammpsThermoStage;
}

const ThermoStageView = ({ stage }: ThermoStageProps): JSX.Element => {
  const columns = stage.columns ?? [];
  const rows = stage.rows ?? [];
  const stepIndex = columns.indexOf("Step");
  const seriesColumns = columns
    .map((name, index) => ({ name, index }))
    .filter(({ index }) => index !== stepIndex);

  if (seriesColumns.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-6 text-center text-sm text-muted-foreground">
        Stage parsed but contains no plottable columns.
      </div>
    );
  }

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      {seriesColumns.map(({ name, index }, paletteIdx) => {
        const lastValue = rows[rows.length - 1]?.[index];
        return (
          <section key={name} className="min-w-0 rounded-md border border-border bg-background p-3">
            <div className="flex items-baseline justify-between gap-3">
              <div className="min-w-0 truncate text-sm font-medium text-foreground">{name}</div>
              <div className="font-mono text-xs text-muted-foreground">
                {Number.isFinite(lastValue) ? lastValue.toPrecision(4) : "—"}
              </div>
            </div>
            <ThermoChart
              stage={stage}
              columnIndex={index}
              color={PALETTE[paletteIdx % PALETTE.length]}
            />
          </section>
        );
      })}
    </div>
  );
};

interface LogPreviewProps {
  projectId: string;
  experimentId: string;
  runId: string;
  file: DiscoveredFile;
}

const LogPreview = ({ projectId, experimentId, runId, file }: LogPreviewProps): JSX.Element => {
  const [response, setResponse] = useState<LammpsLogResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setResponse(null);
    setError(null);

    workspaceApi
      .getRunLammpsLog(projectId, experimentId, runId, file.relPath)
      .then((value) => {
        if (!cancelled) setResponse(value);
      })
      .catch((reason) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : "Failed to load log");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [projectId, experimentId, runId, file.relPath]);

  if (error) {
    return (
      <EmptyState
        icon={<FileText className="h-6 w-6" />}
        title="Cannot read log"
        description={error}
      />
    );
  }

  if (response === null) {
    return <div className="text-sm text-muted-foreground">Loading {file.relPath}…</div>;
  }

  const stages = response.stages ?? [];
  if ((response.nStages ?? 0) === 0 || stages.length === 0) {
    return (
      <EmptyState
        icon={<FileText className="h-6 w-6" />}
        title="No thermo data found"
        description={`molpy parsed ${file.relPath} but found no Per-MPI-rank thermo blocks.`}
      />
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {stages.map((stage, idx) => {
        const firstStep = stage.rows?.[0]?.[stage.columns?.indexOf("Step") ?? -1];
        const stageKey = `${file.relPath}:${Number.isFinite(firstStep) ? firstStep : idx}`;
        return (
          <div key={stageKey} className="flex flex-col gap-2">
            {stages.length > 1 && (
              <div className="text-xs uppercase tracking-wide text-muted-foreground">
                Stage {idx + 1}
              </div>
            )}
            <ThermoStageView stage={stage} />
          </div>
        );
      })}
    </div>
  );
};

interface TrajectoryPanelProps {
  projectId: string;
  experimentId: string;
  runId: string;
  files: DiscoveredFile[];
  active: string | null;
  onSelect: (relPath: string) => void;
}

const TrajectoryPanel = ({
  projectId,
  experimentId,
  runId,
  files,
  active,
  onSelect,
}: TrajectoryPanelProps): JSX.Element => {
  if (files.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-6 text-center text-sm text-muted-foreground">
        No trajectory files discovered.
      </div>
    );
  }

  const activeFile = files.find((file) => file.relPath === active) ?? null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-1 text-xs">
        {files.map((file) => (
          <button
            key={file.relPath}
            type="button"
            onClick={() => onSelect(file.relPath)}
            className={`rounded px-2 py-0.5 transition-colors ${
              active === file.relPath
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            }`}
            title={`${file.relPath} (${formatBytes(file.size)})`}
          >
            {file.name}
          </button>
        ))}
      </div>
      {activeFile ? (
        <TrajectoryViewer
          projectId={projectId}
          experimentId={experimentId}
          runId={runId}
          file={activeFile}
        />
      ) : (
        <div className="text-sm text-muted-foreground">Select a trajectory to render in 3D.</div>
      )}
    </div>
  );
};

export const MolvisTab = ({
  selection,
  snapshot,
  discoveredFiles = [],
}: MolvisTabProps): JSX.Element => {
  const run = useMemo(
    () => snapshot.runs.find((r) => r.id === selection.objectId) ?? null,
    [snapshot.runs, selection.objectId],
  );
  const [activeLog, setActiveLog] = useState<string | null>(null);
  const [activeTrajectory, setActiveTrajectory] = useState<string | null>(null);

  const logFiles = useMemo(() => discoveredFiles.filter(isLogFile), [discoveredFiles]);
  const trajectoryFiles = useMemo(
    () => discoveredFiles.filter((file) => TRAJECTORY_PATTERNS.test(file.name)),
    [discoveredFiles],
  );

  useEffect(() => {
    if (!activeLog && logFiles.length > 0) {
      setActiveLog(logFiles[0].relPath);
    }
  }, [logFiles, activeLog]);

  useEffect(() => {
    if (!activeTrajectory && trajectoryFiles.length > 0) {
      setActiveTrajectory(trajectoryFiles[0].relPath);
    }
  }, [trajectoryFiles, activeTrajectory]);

  if (!run) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          icon={<Atom className="h-6 w-6" />}
          title="Run not found"
          description="The selected run is unavailable."
        />
      </div>
    );
  }

  const activeLogFile = logFiles.find((file) => file.relPath === activeLog) ?? null;

  return (
    <div className="flex-1 overflow-auto bg-background">
      <div className="mx-auto flex max-w-6xl flex-col gap-5 px-4 py-4 md:px-6">
        <div className="flex items-center gap-2">
          <Atom className="h-4 w-4 text-muted-foreground" />
          <div className="text-sm font-medium text-foreground">LAMMPS Output</div>
          <div className="ml-auto text-xs text-muted-foreground">
            {logFiles.length} log · {trajectoryFiles.length} trajectory
          </div>
        </div>

        {logFiles.length > 0 && (
          <OverviewSection title="Thermo (parsed by molpy)">
            <div className="mb-3 flex flex-wrap items-center gap-1 text-xs">
              {logFiles.map((file) => (
                <button
                  key={file.relPath}
                  type="button"
                  onClick={() => setActiveLog(file.relPath)}
                  className={`rounded px-2 py-0.5 transition-colors ${
                    activeLog === file.relPath
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                  }`}
                >
                  {file.relPath}
                </button>
              ))}
            </div>
            {activeLogFile ? (
              <LogPreview
                projectId={run.projectId}
                experimentId={run.experimentId}
                runId={run.id}
                file={activeLogFile}
              />
            ) : (
              <div className="text-sm text-muted-foreground">Select a log to preview.</div>
            )}
          </OverviewSection>
        )}

        <OverviewSection title="Trajectories (rendered by molvis-core)">
          <TrajectoryPanel
            projectId={run.projectId}
            experimentId={run.experimentId}
            runId={run.id}
            files={trajectoryFiles}
            active={activeTrajectory}
            onSelect={setActiveTrajectory}
          />
        </OverviewSection>
      </div>
    </div>
  );
};
