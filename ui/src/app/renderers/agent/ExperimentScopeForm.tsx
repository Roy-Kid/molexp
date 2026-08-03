/**
 * In-bubble form for Plan Mode experiment scope (project + experiment).
 * Submits as `project / experiment` so the existing backend path is unchanged.
 */

import { type JSX, useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { WorkbenchAction } from "@/components/workbench";

export interface ExperimentCatalogEntry {
  project_id: string;
  experiment_id: string;
  label: string;
}

export interface ExperimentScopeFormProps {
  catalog?: ExperimentCatalogEntry[];
  allowCreate?: boolean;
  intro?: string;
  disabled?: boolean;
  onSubmit: (scope: string) => void | Promise<void>;
}

export const ExperimentScopeForm = ({
  catalog = [],
  allowCreate = true,
  intro = "Choose project and experiment for this plan.",
  disabled = false,
  onSubmit,
}: ExperimentScopeFormProps): JSX.Element => {
  const options = useMemo(
    () => catalog.filter((row) => row.project_id && row.experiment_id),
    [catalog],
  );
  const [mode, setMode] = useState<"existing" | "create">(
    options.length > 0 ? "existing" : "create",
  );
  const [selected, setSelected] = useState(options[0]?.label ?? "");
  const [project, setProject] = useState("");
  const [experiment, setExperiment] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit =
    mode === "existing" ? Boolean(selected.trim()) : Boolean(project.trim() && experiment.trim());

  const handleSubmit = async (): Promise<void> => {
    if (!canSubmit || busy || disabled) return;
    setBusy(true);
    setError(null);
    try {
      const scope =
        mode === "existing" ? selected.trim() : `${project.trim()} / ${experiment.trim()}`;
      await onSubmit(scope);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3 rounded-md bg-muted/40 px-3 py-3">
      <p className="text-sm text-foreground">{intro}</p>

      {options.length > 0 && (
        <div className="flex gap-2 text-micro">
          <button
            type="button"
            className={
              mode === "existing"
                ? "rounded-md bg-card px-2 py-1 font-medium text-foreground"
                : "rounded-md px-2 py-1 text-muted-foreground hover:bg-muted"
            }
            onClick={() => setMode("existing")}
            disabled={disabled || busy}
          >
            Existing
          </button>
          {allowCreate ? (
            <button
              type="button"
              className={
                mode === "create"
                  ? "rounded-md bg-card px-2 py-1 font-medium text-foreground"
                  : "rounded-md px-2 py-1 text-muted-foreground hover:bg-muted"
              }
              onClick={() => setMode("create")}
              disabled={disabled || busy}
            >
              Create new
            </button>
          ) : null}
        </div>
      )}

      {mode === "existing" && options.length > 0 ? (
        <div className="space-y-1.5">
          <Label htmlFor="scope-existing" className="text-xs">
            Scope
          </Label>
          <select
            id="scope-existing"
            className="h-9 w-full rounded-md bg-card px-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            disabled={disabled || busy}
          >
            {options.map((row) => (
              <option key={row.label} value={`${row.project_id} / ${row.experiment_id}`}>
                {row.label}
              </option>
            ))}
          </select>
        </div>
      ) : (
        <div className="grid gap-2 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="scope-project" className="text-xs">
              Project
            </Label>
            <Input
              id="scope-project"
              value={project}
              onChange={(e) => setProject(e.target.value)}
              placeholder="my-project"
              disabled={disabled || busy}
              className="border-0 bg-card shadow-none"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="scope-experiment" className="text-xs">
              Experiment
            </Label>
            <Input
              id="scope-experiment"
              value={experiment}
              onChange={(e) => setExperiment(e.target.value)}
              placeholder="my-experiment"
              disabled={disabled || busy}
              className="border-0 bg-card shadow-none"
            />
          </div>
        </div>
      )}

      {error ? <p className="text-xs text-destructive">{error}</p> : null}

      <div className="flex justify-end">
        <WorkbenchAction
          kind="primary"
          size="compact"
          disabled={!canSubmit || busy || disabled}
          onClick={() => void handleSubmit()}
        >
          {busy ? "Starting…" : "Continue"}
        </WorkbenchAction>
      </div>
    </div>
  );
};
