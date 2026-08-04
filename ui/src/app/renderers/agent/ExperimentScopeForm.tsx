/**
 * In-bubble form for Plan Mode experiment scope (project + experiment).
 * Submits as `project / experiment` so the existing backend path is unchanged.
 */

import { type JSX, useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
  const [selected, setSelected] = useState(
    options[0] ? `${options[0].project_id} / ${options[0].experiment_id}` : "",
  );
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
    <div className="space-y-3 rounded-control bg-muted/40 px-3 py-3">
      <p className="text-body-lg text-foreground">{intro}</p>

      {options.length > 0 && (
        <div className="flex gap-2 text-micro">
          <WorkbenchAction
            kind="ghost"
            size="content"
            type="button"
            className={
              mode === "existing"
                ? "rounded-control bg-card px-2 py-1 font-medium text-foreground"
                : "rounded-control px-2 py-1 text-muted-foreground hover:bg-muted"
            }
            onClick={() => setMode("existing")}
            disabled={disabled || busy}
          >
            Existing
          </WorkbenchAction>
          {allowCreate ? (
            <WorkbenchAction
              kind="ghost"
              size="content"
              type="button"
              className={
                mode === "create"
                  ? "rounded-control bg-card px-2 py-1 font-medium text-foreground"
                  : "rounded-control px-2 py-1 text-muted-foreground hover:bg-muted"
              }
              onClick={() => setMode("create")}
              disabled={disabled || busy}
            >
              Create new
            </WorkbenchAction>
          ) : null}
        </div>
      )}

      {mode === "existing" && options.length > 0 ? (
        <div className="space-y-1.5">
          <Label htmlFor="scope-existing" className="text-label">
            Scope
          </Label>
          <Select value={selected} onValueChange={setSelected} disabled={disabled || busy}>
            <SelectTrigger id="scope-existing" className="h-control-comfortable w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {options.map((row) => (
                <SelectItem key={row.label} value={`${row.project_id} / ${row.experiment_id}`}>
                  {row.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      ) : (
        <div className="grid gap-2 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="scope-project" className="text-label">
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
            <Label htmlFor="scope-experiment" className="text-label">
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

      {error ? <p className="text-label text-destructive">{error}</p> : null}

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
