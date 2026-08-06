import { Atom, FileText } from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { EmptyState } from "@/app/components/entity";
import type { LammpsLogResponse, LammpsThermoStage } from "@/app/state/api";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import { Tree, type TreeNodeProps } from "@/components/ui/tree";
import { CHART_SERIES_PALETTE } from "@/lib/chart-tokens";
import { buildFileTree, collectFolderIds } from "@/lib/file-tree";
import { formatBytes } from "@/lib/format-bytes";
import { MolplotLineChart } from "@/plugins/molplot";
import type { DiscoveredFile } from "@/plugins/types";
import { TrajectoryViewer } from "./TrajectoryViewer";

type MolvisTabProps = RendererProps & { discoveredFiles?: DiscoveredFile[] };

const TRAJECTORY_PATTERNS = /\.(lammpstrj|lmptrj|lammpsdump|dump|xyz|extxyz|pdb)$/i;
const LOG_PATTERNS = /(^log\.lammps$|\.lammps\.log$|^lmp\.log$)/i;

const isLogFile = (file: DiscoveredFile): boolean => LOG_PATTERNS.test(file.name);

/** MolRec Zarr group marker (frame / trajectory / meta) — discovery only until a reader ships. */
const isMolrecZarrMarker = (file: DiscoveredFile): boolean => {
  if (file.name.toLowerCase() !== "zarr.json") return false;
  const path = file.relPath.toLowerCase().replace(/\\/g, "/");
  return (
    path.includes("/frame/") ||
    path.includes("/trajectory/") ||
    path.endsWith("/meta/zarr.json") ||
    path.endsWith("/frame/zarr.json") ||
    path.endsWith("/trajectory/zarr.json")
  );
};

interface FileTreeSidebarProps {
  files: DiscoveredFile[];
  active: string | null;
  onSelect: (relPath: string) => void;
}

/**
 * Left-hand file manager. Replaces the earlier dropdown selectors: it shows
 * every molvis-renderable file the run produced (recursively discovered) with
 * its full directory structure, so two same-named files in different
 * execution dirs stay distinguishable.
 */
const FileTreeSidebar = ({ files, active, onSelect }: FileTreeSidebarProps): JSX.Element => {
  const nodes = useMemo(() => buildFileTree(files), [files]);
  const expanded = useMemo(() => collectFolderIds(nodes), [nodes]);

  const renderNode = (node: TreeNodeProps, defaultRender: ReactNode): ReactNode => {
    if (node.kind === "file" && node.path === active) {
      return (
        <div className="bg-accent-muted text-accent-muted-foreground [&_span]:text-accent-muted-foreground">
          {defaultRender}
        </div>
      );
    }
    return defaultRender;
  };

  return (
    <aside className="flex w-60 flex-none flex-col bg-surface/45">
      <div className="flex h-control-comfortable items-center gap-2 px-3">
        <Atom className="h-4 w-4 text-accent" />
        <span className="text-label font-medium text-foreground">MolVis</span>
        <span className="ml-auto text-micro text-muted-foreground">{files.length}</span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-2">
        <Tree
          nodes={nodes}
          defaultExpandedIds={expanded}
          onSelect={(node) => {
            if (node.kind === "file") {
              onSelect(node.path);
            }
          }}
          renderNode={renderNode}
        />
      </div>
    </aside>
  );
};

interface ThermoChartProps {
  stage: LammpsThermoStage;
  columnIndex: number;
  color: string;
}

const ThermoChart = ({ stage, columnIndex, color }: ThermoChartProps): JSX.Element => {
  // Defaults live inside useMemo so a missing ``stage.columns``/``rows``
  // doesn't materialise a fresh ``[]`` per render and invalidate the
  // memo, which would tear down + re-mount the plotly chart on every
  // parent update.
  const config = useMemo(() => {
    const columns = stage.columns ?? [];
    const rows = stage.rows ?? [];
    const stepIndex = columns.indexOf("Step");
    return {
      series: [
        {
          id: columns[columnIndex] ?? `col${columnIndex}`,
          label: columns[columnIndex],
          color,
          initialPoints: rows.map((row, idx) => ({
            x: stepIndex >= 0 ? row[stepIndex] : idx,
            y: row[columnIndex],
          })),
        },
      ],
      xAxis: { label: "Step" },
      hovertemplate: "%{y:.6g}<extra></extra>",
      hovermode: "x unified" as const,
      modebar: true,
      modebarRemove: ["lasso2d", "select2d", "toggleSpikelines"],
      theme: "auto" as const,
    };
  }, [color, stage, columnIndex]);

  return (
    <MolplotLineChart
      config={config}
      style={{ width: "100%", height: "var(--spacing-chart-md)" }}
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
      <div className="bg-surface/60 px-3 py-6 text-center text-body-lg text-muted-foreground">
        Stage parsed but contains no plottable columns.
      </div>
    );
  }

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      {seriesColumns.map(({ name, index }, paletteIdx) => {
        const lastValue = rows[rows.length - 1]?.[index];
        return (
          <section key={name} className="min-w-0 bg-surface/65 p-3">
            <div className="flex items-baseline justify-between gap-3">
              <div className="min-w-0 truncate text-body-lg font-medium text-foreground">
                {name}
              </div>
              <div className="font-mono text-label text-muted-foreground">
                {Number.isFinite(lastValue) ? lastValue.toPrecision(4) : "—"}
              </div>
            </div>
            <ThermoChart
              stage={stage}
              columnIndex={index}
              color={CHART_SERIES_PALETTE[paletteIdx % CHART_SERIES_PALETTE.length]}
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
    return <div className="text-body-lg text-muted-foreground">Loading {file.relPath}…</div>;
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
              <div className="text-label uppercase tracking-wide text-muted-foreground">
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

interface PreviewPaneProps {
  projectId: string;
  experimentId: string;
  runId: string;
  file: DiscoveredFile | null;
}

/**
 * Right-hand content area. Fills the blank space (with padding) and renders the
 * selected file: thermo charts for LAMMPS logs, a space-filling 3D canvas for
 * trajectories, a friendly notice for anything molvis cannot draw.
 */
const PreviewPane = ({ projectId, experimentId, runId, file }: PreviewPaneProps): JSX.Element => {
  if (!file) {
    return (
      <div className="flex flex-1 items-center justify-center p-6">
        <EmptyState
          icon={<Atom className="h-6 w-6" />}
          title="Select a file"
          description="Pick a log or trajectory from the file tree to preview it here."
        />
      </div>
    );
  }

  const header = (
    <div className="flex flex-none items-center gap-2 px-4 pt-4 text-label text-muted-foreground">
      <span className="truncate font-mono text-foreground" title={file.relPath}>
        {file.relPath}
      </span>
      <span className="ml-auto flex-none tabular-nums">{formatBytes(file.size)}</span>
    </div>
  );

  if (isLogFile(file)) {
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        {header}
        <div className="min-h-0 flex-1 overflow-auto p-4">
          <LogPreview projectId={projectId} experimentId={experimentId} runId={runId} file={file} />
        </div>
      </div>
    );
  }

  if (TRAJECTORY_PATTERNS.test(file.name)) {
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        {header}
        <div className="min-h-0 flex-1 p-4">
          <TrajectoryViewer
            projectId={projectId}
            experimentId={experimentId}
            runId={runId}
            file={file}
            className="h-full"
          />
        </div>
      </div>
    );
  }

  if (isMolrecZarrMarker(file)) {
    return (
      <div className="flex flex-1 items-center justify-center p-6">
        <EmptyState
          icon={<Atom className="h-6 w-6" />}
          title="MolRec Zarr package"
          description="This looks like a MolRec Zarr root (frame/trajectory/meta). In-browser MolRec frame streaming is not wired yet — export classic formats (xyz/dump) or use molrs until the Zarr reader lands."
        />
      </div>
    );
  }

  return (
    <div className="flex flex-1 items-center justify-center p-6">
      <EmptyState
        icon={<FileText className="h-6 w-6" />}
        title="No molvis preview"
        description={`${file.name} was discovered but has no log or trajectory renderer.`}
      />
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
  const [activeFile, setActiveFile] = useState<string | null>(null);

  // Default selection: first trajectory (the headline view), else first log,
  // else first discovered file — recomputed only while nothing is selected.
  const defaultRelPath = useMemo(() => {
    const trajectory = discoveredFiles.find((file) => TRAJECTORY_PATTERNS.test(file.name));
    const log = discoveredFiles.find(isLogFile);
    return (trajectory ?? log ?? discoveredFiles[0])?.relPath ?? null;
  }, [discoveredFiles]);

  useEffect(() => {
    if (!activeFile && defaultRelPath) {
      setActiveFile(defaultRelPath);
    }
  }, [activeFile, defaultRelPath]);

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

  const selectedFile = discoveredFiles.find((file) => file.relPath === activeFile) ?? null;

  return (
    <div className="flex min-h-0 flex-1 bg-background">
      <FileTreeSidebar files={discoveredFiles} active={activeFile} onSelect={setActiveFile} />
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <PreviewPane
          projectId={run.projectId}
          experimentId={run.experimentId}
          runId={run.id}
          file={selectedFile}
        />
      </main>
    </div>
  );
};
