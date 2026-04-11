/**
 * Snapshot & Diff components.
 *
 * Architecture:
 *   - Run    → fixed snapshot   → <RunSnapshotPanel>   (read-only list)
 *   - Experiment → compare runs → <SnapshotDiffPanel>  (pick two runs, show diff)
 *
 * Both panels are exported for use in RunViewer and ExperimentViewer respectively.
 */

import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Code,
  Database,
  GitCompareArrows,
  Hash,
  Settings,
} from "lucide-react";
import { useMemo, useState } from "react";
import type { TaskSnapshotResponse } from "@/api/generated/models/TaskSnapshotResponse";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// ============================================================================
// Types
// ============================================================================

type FieldStatus = "unchanged" | "modified" | "added" | "removed";

interface TaskDiff {
  taskId: string;
  taskType: string;
  status: FieldStatus;
  codeChanged: boolean;
  configChanged: boolean;
  oldSnapshot: TaskSnapshotResponse | null;
  newSnapshot: TaskSnapshotResponse | null;
}

/** A snapshot set attached to a single run. */
export interface RunSnapshotSet {
  runId: string;
  runLabel: string;
  createdAt: string;
  snapshots: TaskSnapshotResponse[];
}

// ============================================================================
// Mock data
// ============================================================================

const MOCK_RUN_SNAPSHOTS: Record<string, RunSnapshotSet> = {
  "run-001": {
    runId: "run-001",
    runLabel: "run-001 (latest)",
    createdAt: "2026-02-13T08:30:00Z",
    snapshots: [
      {
        taskId: "PrepareData_a1b2c3d4",
        taskType: "PrepareData",
        codeHash: "e7f3a09b16c2d481",
        configHash: "3bc1f84e927da056",
        codeSource: `def execute(self, inputs: dict) -> dict:
    raw_path = Path(self.config.input_dir)
    df = pd.read_parquet(raw_path / "molecules.parquet")
    if self.config.shuffle:
        df = df.sample(frac=1, random_state=42)
    train, test = train_test_split(df, test_size=0.2)
    return {"train": train, "test": test}`,
        snapshotKey: "e7f3a09b16c2d481:3bc1f84e927da056",
        createdAt: "2026-02-13T08:30:00Z",
        configData: { input_dir: "/data/raw", format: "parquet", shuffle: true },
      },
      {
        taskId: "TrainModel_e5f6a7b8",
        taskType: "TrainModel",
        codeHash: "a20c4f1d88e53b79",
        configHash: "71d0e2ca5f38b614",
        codeSource: `def execute(self, inputs: dict) -> dict:
    train_data = inputs["train"]
    model = GNNModel(version="v2", hidden_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
    for epoch in range(self.config.epochs):
        loss = train_epoch(model, train_data, optimizer, self.config.batch_size)
        scheduler.step()
        if epoch % 50 == 0:
            logger.info(f"Epoch {epoch}: loss={loss:.4f}")
    return {"model": model, "final_loss": loss}`,
        snapshotKey: "a20c4f1d88e53b79:71d0e2ca5f38b614",
        createdAt: "2026-02-13T08:30:00Z",
        configData: {
          model: "gnn-v2",
          epochs: 200,
          lr: 0.0005,
          batch_size: 64,
          scheduler: "cosine",
        },
      },
      {
        taskId: "Evaluate_c9d0e1f2",
        taskType: "Evaluate",
        codeHash: "5b8d12ef6ca0793e",
        configHash: "d4a176c023be8f51",
        codeSource: `def execute(self, inputs: dict) -> dict:
    model = inputs["model"]
    test_data = inputs["test"]
    predictions = model.predict(test_data)
    results = {}
    for metric in self.config.metrics:
        results[metric] = compute_metric(metric, test_data.y, predictions)
    return {"metrics": results}`,
        snapshotKey: "5b8d12ef6ca0793e:d4a176c023be8f51",
        createdAt: "2026-02-13T08:30:00Z",
        configData: { metrics: ["mae", "rmse", "r2"], split: "test" },
      },
      {
        taskId: "ExportReport_f3a4b5c6",
        taskType: "ExportReport",
        codeHash: "1c9e3d7f42b08a65",
        configHash: "9fe2b8a04d61c73f",
        codeSource: `def execute(self, inputs: dict) -> dict:
    metrics = inputs["metrics"]
    report = ReportBuilder(format=self.config.output_format)
    report.add_section("Metrics", metrics)
    if self.config.include_plots:
        report.add_plots(generate_metric_plots(metrics))
    output_path = report.save()
    return {"report_path": str(output_path)}`,
        snapshotKey: "1c9e3d7f42b08a65:9fe2b8a04d61c73f",
        createdAt: "2026-02-13T08:30:00Z",
        configData: { output_format: "html", include_plots: true },
      },
    ],
  },
  "run-002": {
    runId: "run-002",
    runLabel: "run-002",
    createdAt: "2026-02-12T14:00:00Z",
    snapshots: [
      {
        taskId: "PrepareData_a1b2c3d4",
        taskType: "PrepareData",
        codeHash: "e7f3a09b16c2d481",
        configHash: "3bc1f84e927da056",
        codeSource: `def execute(self, inputs: dict) -> dict:
    raw_path = Path(self.config.input_dir)
    df = pd.read_parquet(raw_path / "molecules.parquet")
    if self.config.shuffle:
        df = df.sample(frac=1, random_state=42)
    train, test = train_test_split(df, test_size=0.2)
    return {"train": train, "test": test}`,
        snapshotKey: "e7f3a09b16c2d481:3bc1f84e927da056",
        createdAt: "2026-02-12T14:00:00Z",
        configData: { input_dir: "/data/raw", format: "parquet", shuffle: true },
      },
      {
        taskId: "TrainModel_e5f6a7b8",
        taskType: "TrainModel",
        codeHash: "f91ab32c07d64e18",
        configHash: "c5249e8b1fa07d63",
        codeSource: `def execute(self, inputs: dict) -> dict:
    train_data = inputs["train"]
    model = GNNModel(version="v1")
    optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
    for epoch in range(self.config.epochs):
        loss = train_epoch(model, train_data, optimizer, self.config.batch_size)
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: loss={loss:.4f}")
    return {"model": model, "final_loss": loss}`,
        snapshotKey: "f91ab32c07d64e18:c5249e8b1fa07d63",
        createdAt: "2026-02-12T14:00:00Z",
        configData: {
          model: "gnn-v1",
          epochs: 100,
          lr: 0.001,
          batch_size: 32,
        },
      },
      {
        taskId: "Evaluate_c9d0e1f2",
        taskType: "Evaluate",
        codeHash: "5b8d12ef6ca0793e",
        configHash: "88c1d2e3f4a5b607",
        codeSource: `def execute(self, inputs: dict) -> dict:
    model = inputs["model"]
    test_data = inputs["test"]
    predictions = model.predict(test_data)
    results = {}
    for metric in self.config.metrics:
        results[metric] = compute_metric(metric, test_data.y, predictions)
    return {"metrics": results}`,
        snapshotKey: "5b8d12ef6ca0793e:88c1d2e3f4a5b607",
        createdAt: "2026-02-12T14:00:00Z",
        configData: { metrics: ["mae", "rmse"], split: "test" },
      },
    ],
  },
  "run-003": {
    runId: "run-003",
    runLabel: "run-003 (baseline)",
    createdAt: "2026-02-11T10:00:00Z",
    snapshots: [
      {
        taskId: "PrepareData_a1b2c3d4",
        taskType: "PrepareData",
        codeHash: "e7f3a09b16c2d481",
        configHash: "3bc1f84e927da056",
        codeSource: `def execute(self, inputs: dict) -> dict:
    raw_path = Path(self.config.input_dir)
    df = pd.read_parquet(raw_path / "molecules.parquet")
    if self.config.shuffle:
        df = df.sample(frac=1, random_state=42)
    train, test = train_test_split(df, test_size=0.2)
    return {"train": train, "test": test}`,
        snapshotKey: "e7f3a09b16c2d481:3bc1f84e927da056",
        createdAt: "2026-02-11T10:00:00Z",
        configData: { input_dir: "/data/raw", format: "parquet", shuffle: true },
      },
      {
        taskId: "TrainModel_e5f6a7b8",
        taskType: "TrainModel",
        codeHash: "f91ab32c07d64e18",
        configHash: "c5249e8b1fa07d63",
        codeSource: `def execute(self, inputs: dict) -> dict:
    train_data = inputs["train"]
    model = GNNModel(version="v1")
    optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
    for epoch in range(self.config.epochs):
        loss = train_epoch(model, train_data, optimizer, self.config.batch_size)
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: loss={loss:.4f}")
    return {"model": model, "final_loss": loss}`,
        snapshotKey: "f91ab32c07d64e18:c5249e8b1fa07d63",
        createdAt: "2026-02-11T10:00:00Z",
        configData: {
          model: "gnn-v1",
          epochs: 100,
          lr: 0.001,
          batch_size: 32,
        },
      },
      {
        taskId: "Evaluate_c9d0e1f2",
        taskType: "Evaluate",
        codeHash: "5b8d12ef6ca0793e",
        configHash: "88c1d2e3f4a5b607",
        codeSource: `def execute(self, inputs: dict) -> dict:
    model = inputs["model"]
    test_data = inputs["test"]
    predictions = model.predict(test_data)
    results = {}
    for metric in self.config.metrics:
        results[metric] = compute_metric(metric, test_data.y, predictions)
    return {"metrics": results}`,
        snapshotKey: "5b8d12ef6ca0793e:88c1d2e3f4a5b607",
        createdAt: "2026-02-11T10:00:00Z",
        configData: { metrics: ["mae", "rmse"], split: "test" },
      },
    ],
  },
};

/** Look up mock snapshot set for a run. Falls back to empty. */
export function getMockRunSnapshots(runId: string): RunSnapshotSet | null {
  return MOCK_RUN_SNAPSHOTS[runId] ?? null;
}

/** All available mock run IDs (for the experiment-level selector). */
export function getMockRunIds(): string[] {
  return Object.keys(MOCK_RUN_SNAPSHOTS);
}

/** All mock run snapshot sets (for experiment diff). */
export function getMockAllRunSnapshots(): RunSnapshotSet[] {
  return Object.values(MOCK_RUN_SNAPSHOTS);
}

// ============================================================================
// Diff computation
// ============================================================================

function computeDiff(
  oldSnapshots: TaskSnapshotResponse[],
  newSnapshots: TaskSnapshotResponse[],
): TaskDiff[] {
  const oldMap = new Map(oldSnapshots.map((s) => [s.taskId, s]));
  const newMap = new Map(newSnapshots.map((s) => [s.taskId, s]));
  const allIds = new Set([...oldMap.keys(), ...newMap.keys()]);

  const diffs: TaskDiff[] = [];
  for (const id of allIds) {
    const oldSnap = oldMap.get(id) ?? null;
    const newSnap = newMap.get(id) ?? null;

    if (!oldSnap && newSnap) {
      diffs.push({
        taskId: id,
        taskType: newSnap.taskType,
        status: "added",
        codeChanged: true,
        configChanged: true,
        oldSnapshot: null,
        newSnapshot: newSnap,
      });
    } else if (oldSnap && !newSnap) {
      diffs.push({
        taskId: id,
        taskType: oldSnap.taskType,
        status: "removed",
        codeChanged: true,
        configChanged: true,
        oldSnapshot: oldSnap,
        newSnapshot: null,
      });
    } else if (oldSnap && newSnap) {
      const codeChanged = oldSnap.codeHash !== newSnap.codeHash;
      const configChanged = oldSnap.configHash !== newSnap.configHash;
      diffs.push({
        taskId: id,
        taskType: newSnap.taskType,
        status: codeChanged || configChanged ? "modified" : "unchanged",
        codeChanged,
        configChanged,
        oldSnapshot: oldSnap,
        newSnapshot: newSnap,
      });
    }
  }

  const order: Record<FieldStatus, number> = {
    modified: 0,
    added: 1,
    removed: 2,
    unchanged: 3,
  };
  diffs.sort((a, b) => order[a.status] - order[b.status]);
  return diffs;
}

// ============================================================================
// Shared sub-components
// ============================================================================

const StatusBadge = ({ status }: { status: FieldStatus }): JSX.Element => {
  const styles: Record<FieldStatus, string> = {
    unchanged: "bg-emerald-500/10 text-emerald-700 border-emerald-500/20",
    modified: "bg-amber-500/10 text-amber-700 border-amber-500/20",
    added: "bg-blue-500/10 text-blue-700 border-blue-500/20",
    removed: "bg-red-500/10 text-red-700 border-red-500/20",
  };
  return (
    <Badge variant="outline" className={`text-xs ${styles[status]}`}>
      {status}
    </Badge>
  );
};

const ConfigValue = ({
  label,
  oldVal,
  newVal,
}: {
  label: string;
  oldVal: string | undefined;
  newVal: string | undefined;
}): JSX.Element | null => {
  if (oldVal === undefined && newVal === undefined) return null;
  const changed = oldVal !== newVal;
  return (
    <div className="flex items-start gap-2 text-xs leading-relaxed">
      <span className="text-muted-foreground min-w-[100px] shrink-0">{label}</span>
      {changed ? (
        <span className="font-mono">
          <span className="line-through text-red-500/70">{oldVal ?? "—"}</span>
          <span className="mx-1.5 text-muted-foreground">&rarr;</span>
          <span className="text-emerald-600 font-semibold">{newVal ?? "—"}</span>
        </span>
      ) : (
        <span className="font-mono text-foreground">{newVal ?? "—"}</span>
      )}
    </div>
  );
};

// ============================================================================
// SnapshotCard — single task snapshot (used in RunSnapshotPanel)
// ============================================================================

export const SnapshotCard = ({ snapshot }: { snapshot: TaskSnapshotResponse }): JSX.Element => {
  const [codeExpanded, setCodeExpanded] = useState(false);

  return (
    <div className="rounded-md border p-4 bg-card space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Hash className="h-4 w-4 text-muted-foreground" />
          <span className="font-mono text-sm font-medium">{snapshot.taskId}</span>
        </div>
        <Badge variant="outline" className="text-xs">
          {snapshot.taskType}
        </Badge>
      </div>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div>
          <p className="text-muted-foreground uppercase mb-1">Code Hash</p>
          <p className="font-mono truncate" title={snapshot.codeHash}>
            {snapshot.codeHash}
          </p>
        </div>
        <div>
          <p className="text-muted-foreground uppercase mb-1">Config Hash</p>
          <p className="font-mono truncate" title={snapshot.configHash}>
            {snapshot.configHash}
          </p>
        </div>
        <div className="col-span-2">
          <p className="text-muted-foreground uppercase mb-1">Snapshot Key</p>
          <p className="font-mono text-xs truncate" title={snapshot.snapshotKey}>
            {snapshot.snapshotKey}
          </p>
        </div>
      </div>
      {snapshot.configData && Object.keys(snapshot.configData).length > 0 && (
        <div>
          <p className="text-xs text-muted-foreground uppercase mb-1">Config</p>
          <pre className="text-xs font-mono bg-muted/30 rounded p-2 overflow-x-auto max-h-32">
            {JSON.stringify(snapshot.configData, null, 2)}
          </pre>
        </div>
      )}
      {snapshot.codeSource && (
        <div>
          <button
            type="button"
            className="flex items-center gap-1 text-xs text-muted-foreground uppercase mb-1 hover:text-foreground transition-colors cursor-pointer"
            onClick={() => setCodeExpanded(!codeExpanded)}
          >
            {codeExpanded ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
            Source Code
          </button>
          {codeExpanded && (
            <pre className="text-xs font-mono bg-slate-950 text-slate-50 rounded p-3 overflow-x-auto max-h-64">
              {snapshot.codeSource}
            </pre>
          )}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// CodeDiffBlock — line-by-line code diff
// ============================================================================

interface DiffLine {
  type: "same" | "added" | "removed";
  text: string;
}

function computeLineDiff(oldCode: string, newCode: string): DiffLine[] {
  const oldLines = oldCode.split("\n");
  const newLines = newCode.split("\n");

  // Simple LCS-based diff
  const m = oldLines.length;
  const n = newLines.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        oldLines[i - 1] === newLines[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  const result: DiffLine[] = [];
  let i = m,
    j = n;
  const stack: DiffLine[] = [];
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oldLines[i - 1] === newLines[j - 1]) {
      stack.push({ type: "same", text: oldLines[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      stack.push({ type: "added", text: newLines[j - 1] });
      j--;
    } else {
      stack.push({ type: "removed", text: oldLines[i - 1] });
      i--;
    }
  }
  stack.reverse();
  result.push(...stack);
  return result;
}

const CodeDiffBlock = ({ oldCode, newCode }: { oldCode: string; newCode: string }): JSX.Element => {
  const lines = useMemo(() => computeLineDiff(oldCode, newCode), [oldCode, newCode]);

  return (
    <pre className="text-xs font-mono rounded overflow-x-auto max-h-72 bg-slate-950 p-3">
      {lines.map((line) => {
        const bgClass =
          line.type === "removed"
            ? "bg-red-500/15 text-red-300"
            : line.type === "added"
              ? "bg-emerald-500/15 text-emerald-300"
              : "text-slate-400";
        const prefix = line.type === "removed" ? "- " : line.type === "added" ? "+ " : "  ";
        return (
          <div
            key={`diff-line-${line.type}-${line.text.slice(0, 40)}`}
            className={`${bgClass} px-2 leading-relaxed`}
          >
            <span className="select-none opacity-50 mr-2">{prefix}</span>
            {line.text}
          </div>
        );
      })}
    </pre>
  );
};

// ============================================================================
// DiffCard — single task diff (used in SnapshotDiffPanel)
// ============================================================================

const DiffCard = ({ diff }: { diff: TaskDiff }): JSX.Element => {
  const oldCfg = diff.oldSnapshot?.configData ?? {};
  const newCfg = diff.newSnapshot?.configData ?? {};
  const allKeys = Array.from(new Set([...Object.keys(oldCfg), ...Object.keys(newCfg)]));

  return (
    <div className="rounded-md border p-4 bg-card space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Hash className="h-4 w-4 text-muted-foreground" />
          <span className="font-mono text-sm font-medium">{diff.taskId}</span>
          <Badge variant="outline" className="text-xs">
            {diff.taskType}
          </Badge>
        </div>
        <StatusBadge status={diff.status} />
      </div>

      {/* Change indicators */}
      {diff.status !== "unchanged" && (
        <div className="flex items-center gap-3 text-xs">
          <span
            className={`flex items-center gap-1 ${
              diff.codeChanged ? "text-amber-600" : "text-muted-foreground"
            }`}
          >
            <Code className="h-3.5 w-3.5" />
            code {diff.codeChanged ? "changed" : "same"}
          </span>
          <span
            className={`flex items-center gap-1 ${
              diff.configChanged ? "text-amber-600" : "text-muted-foreground"
            }`}
          >
            <Settings className="h-3.5 w-3.5" />
            config {diff.configChanged ? "changed" : "same"}
          </span>
        </div>
      )}

      {/* Hash diff */}
      {diff.status === "modified" && (
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <p className="text-muted-foreground uppercase mb-1">Code Hash</p>
            <p className="font-mono">
              {diff.codeChanged ? (
                <>
                  <span className="line-through text-red-500/70">{diff.oldSnapshot?.codeHash}</span>
                  <br />
                  <span className="text-emerald-600 font-semibold">
                    {diff.newSnapshot?.codeHash}
                  </span>
                </>
              ) : (
                <span>{diff.newSnapshot?.codeHash}</span>
              )}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground uppercase mb-1">Config Hash</p>
            <p className="font-mono">
              {diff.configChanged ? (
                <>
                  <span className="line-through text-red-500/70">
                    {diff.oldSnapshot?.configHash}
                  </span>
                  <br />
                  <span className="text-emerald-600 font-semibold">
                    {diff.newSnapshot?.configHash}
                  </span>
                </>
              ) : (
                <span>{diff.newSnapshot?.configHash}</span>
              )}
            </p>
          </div>
        </div>
      )}

      {/* Config field-level diff */}
      {diff.status === "modified" && diff.configChanged && allKeys.length > 0 && (
        <div className="border-t pt-3 space-y-1.5">
          <p className="text-xs text-muted-foreground uppercase mb-2">Config Diff</p>
          {allKeys.map((key) => (
            <ConfigValue
              key={key}
              label={key}
              oldVal={oldCfg[key] !== undefined ? JSON.stringify(oldCfg[key]) : undefined}
              newVal={newCfg[key] !== undefined ? JSON.stringify(newCfg[key]) : undefined}
            />
          ))}
        </div>
      )}

      {/* Code diff */}
      {diff.status === "modified" &&
        diff.codeChanged &&
        diff.oldSnapshot?.codeSource &&
        diff.newSnapshot?.codeSource && (
          <div className="border-t pt-3">
            <p className="text-xs text-muted-foreground uppercase mb-2">Code Diff</p>
            <CodeDiffBlock
              oldCode={diff.oldSnapshot.codeSource}
              newCode={diff.newSnapshot.codeSource}
            />
          </div>
        )}

      {/* Added task */}
      {diff.status === "added" && diff.newSnapshot?.configData && (
        <div>
          <p className="text-xs text-muted-foreground uppercase mb-1">Config (new)</p>
          <pre className="text-xs font-mono bg-emerald-500/5 border border-emerald-500/20 rounded p-2 overflow-x-auto max-h-32">
            {JSON.stringify(diff.newSnapshot.configData, null, 2)}
          </pre>
        </div>
      )}
      {diff.status === "added" && diff.newSnapshot?.codeSource && (
        <div>
          <p className="text-xs text-muted-foreground uppercase mb-1">Source Code (new)</p>
          <pre className="text-xs font-mono bg-emerald-500/5 border border-emerald-500/20 rounded p-3 overflow-x-auto max-h-48 text-emerald-800">
            {diff.newSnapshot.codeSource}
          </pre>
        </div>
      )}

      {/* Removed task */}
      {diff.status === "removed" && diff.oldSnapshot?.configData && (
        <div>
          <p className="text-xs text-muted-foreground uppercase mb-1">Config (removed)</p>
          <pre className="text-xs font-mono bg-red-500/5 border border-red-500/20 rounded p-2 overflow-x-auto max-h-32 line-through">
            {JSON.stringify(diff.oldSnapshot.configData, null, 2)}
          </pre>
        </div>
      )}
      {diff.status === "removed" && diff.oldSnapshot?.codeSource && (
        <div>
          <p className="text-xs text-muted-foreground uppercase mb-1">Source Code (removed)</p>
          <pre className="text-xs font-mono bg-red-500/5 border border-red-500/20 rounded p-3 overflow-x-auto max-h-48 line-through text-red-800">
            {diff.oldSnapshot.codeSource}
          </pre>
        </div>
      )}

      {/* Unchanged */}
      {diff.status === "unchanged" && (
        <p className="text-xs text-muted-foreground italic">
          No changes — snapshot key{" "}
          <span className="font-mono">{diff.newSnapshot?.snapshotKey}</span>
        </p>
      )}
    </div>
  );
};

// ============================================================================
// DiffSummaryBar
// ============================================================================

const DiffSummaryBar = ({ diffs }: { diffs: TaskDiff[] }): JSX.Element => {
  const counts = useMemo(() => {
    const c = { unchanged: 0, modified: 0, added: 0, removed: 0 };
    for (const d of diffs) c[d.status]++;
    return c;
  }, [diffs]);

  return (
    <div className="flex items-center gap-4 text-xs">
      {counts.modified > 0 && (
        <span className="flex items-center gap-1 text-amber-600">
          <AlertCircle className="h-3.5 w-3.5" />
          {counts.modified} modified
        </span>
      )}
      {counts.added > 0 && (
        <span className="flex items-center gap-1 text-blue-600">+ {counts.added} added</span>
      )}
      {counts.removed > 0 && (
        <span className="flex items-center gap-1 text-red-600">- {counts.removed} removed</span>
      )}
      {counts.unchanged > 0 && (
        <span className="flex items-center gap-1 text-emerald-600">
          <CheckCircle2 className="h-3.5 w-3.5" />
          {counts.unchanged} unchanged
        </span>
      )}
    </div>
  );
};

// ============================================================================
// RunSnapshotPanel — for RunViewer "Snapshot" tab
// ============================================================================

/**
 * Displays the fixed snapshot set for a single run.
 * This is read-only — a run's snapshot is immutable.
 */
export const RunSnapshotPanel = ({ runId }: { runId: string }): JSX.Element => {
  // In production: fetch from API.  For now: mock data.
  const snapshotSet = getMockRunSnapshots(runId);

  if (!snapshotSet) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
        <Database className="h-10 w-10 opacity-20" />
        <p className="text-sm">No snapshot recorded for this run.</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div className="px-6 py-4 border-b bg-background flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Hash className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">
            Task Snapshots
            <span className="text-muted-foreground font-normal ml-1">
              ({snapshotSet.snapshots.length})
            </span>
          </span>
        </div>
        <span className="text-xs text-muted-foreground font-mono">
          {new Date(snapshotSet.createdAt).toLocaleString()}
        </span>
      </div>
      <div className="flex-1 p-6 space-y-3">
        {snapshotSet.snapshots.map((snap) => (
          <SnapshotCard key={snap.taskId} snapshot={snap} />
        ))}
      </div>
    </div>
  );
};

// ============================================================================
// SnapshotDiffPanel — for ExperimentViewer "Diff" tab
// ============================================================================

/**
 * Lets the user pick two runs and shows a field-level diff of their snapshots.
 */
export const SnapshotDiffPanel = ({
  experimentRunIds,
}: {
  experimentRunIds: string[];
}): JSX.Element => {
  const allSets = getMockAllRunSnapshots();
  // Filter to runs that belong to this experiment (mock: use all available)
  const availableSets = allSets.filter(
    (s) => experimentRunIds.length === 0 || experimentRunIds.includes(s.runId),
  );

  const [baseId, setBaseId] = useState<string>(availableSets[1]?.runId ?? "");
  const [targetId, setTargetId] = useState<string>(availableSets[0]?.runId ?? "");

  const baseSet = availableSets.find((s) => s.runId === baseId);
  const targetSet = availableSets.find((s) => s.runId === targetId);

  const diffs = useMemo(() => {
    if (!baseSet || !targetSet) return [];
    return computeDiff(baseSet.snapshots, targetSet.snapshots);
  }, [baseSet, targetSet]);

  const ready = baseSet && targetSet && baseId !== targetId;

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      {/* Run selector */}
      <div className="px-6 py-4 border-b bg-background">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <GitCompareArrows className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Compare</span>
          </div>

          <Select value={baseId} onValueChange={setBaseId}>
            <SelectTrigger size="sm">
              <SelectValue placeholder="Base run" />
            </SelectTrigger>
            <SelectContent>
              {availableSets.map((s) => (
                <SelectItem key={s.runId} value={s.runId}>
                  {s.runLabel}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <span className="text-muted-foreground text-sm">&rarr;</span>

          <Select value={targetId} onValueChange={setTargetId}>
            <SelectTrigger size="sm">
              <SelectValue placeholder="Target run" />
            </SelectTrigger>
            <SelectContent>
              {availableSets.map((s) => (
                <SelectItem key={s.runId} value={s.runId}>
                  {s.runLabel}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {ready && <DiffSummaryBar diffs={diffs} />}
        </div>
        {baseId === targetId && baseId !== "" && (
          <p className="text-xs text-amber-600 mt-2">
            Base and target are the same run — select different runs to see a diff.
          </p>
        )}
      </div>

      {/* Diff cards */}
      <div className="flex-1 p-6 space-y-3">
        {!ready ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
            <GitCompareArrows className="h-10 w-10 opacity-20" />
            <p className="text-sm">Select two different runs to compare their snapshots.</p>
          </div>
        ) : diffs.length === 0 ? (
          <p className="text-sm text-muted-foreground">No tasks to compare.</p>
        ) : (
          diffs.map((diff) => <DiffCard key={diff.taskId} diff={diff} />)
        )}
      </div>
    </div>
  );
};
