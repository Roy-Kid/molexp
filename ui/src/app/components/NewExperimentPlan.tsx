import { type JSX, useMemo, useState } from "react";
import { PlanComposer } from "@/app/components/PlanComposer";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { WorkspaceSnapshot } from "@/app/types";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/**
 * Inline "plan an experiment" composer for the new-task view. Picks a target
 * experiment, runs PlanMode, and on completion opens the plan's SESSION — plan
 * generation is just another way to start an agent session, so it lives inside
 * the new-task surface rather than a separate page.
 */
export const NewExperimentPlan = ({
  snapshot,
  onRefresh,
}: {
  snapshot: WorkspaceSnapshot;
  onRefresh: () => void;
}): JSX.Element => {
  const nav = useNavigationState(snapshot);
  const projects = snapshot.projects;
  const [projectId, setProjectId] = useState<string>(projects[0]?.id ?? "");

  const experiments = useMemo(
    () => snapshot.experiments.filter((e) => e.projectId === projectId),
    [snapshot.experiments, projectId],
  );
  const [experimentId, setExperimentId] = useState<string>("");
  const effectiveExperimentId = experiments.some((e) => e.id === experimentId)
    ? experimentId
    : (experiments[0]?.id ?? "");

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1">
          <label htmlFor="plan-project" className="text-xs font-medium text-muted-foreground">
            Project
          </label>
          <Select
            value={projectId}
            onValueChange={(v) => {
              setProjectId(v);
              setExperimentId("");
            }}
          >
            <SelectTrigger id="plan-project">
              <SelectValue placeholder="Select a project" />
            </SelectTrigger>
            <SelectContent>
              {projects.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <label htmlFor="plan-experiment" className="text-xs font-medium text-muted-foreground">
            Experiment
          </label>
          <Select
            value={effectiveExperimentId}
            onValueChange={setExperimentId}
            disabled={experiments.length === 0}
          >
            <SelectTrigger id="plan-experiment">
              <SelectValue placeholder="Select an experiment" />
            </SelectTrigger>
            <SelectContent>
              {experiments.map((e) => (
                <SelectItem key={e.id} value={e.id}>
                  {e.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {effectiveExperimentId ? (
        <PlanComposer
          key={`${projectId}:${effectiveExperimentId}`}
          projectId={projectId}
          experimentId={effectiveExperimentId}
          onPlanComplete={(task) => {
            onRefresh();
            // Open the freshly-generated plan's session.
            nav.setSelection({ objectType: "agent", objectId: task.taskId });
          }}
        />
      ) : (
        <p className="text-sm italic text-muted-foreground">
          {projects.length === 0
            ? "Create a project and an experiment first, then come back to plan one."
            : "Select an experiment to plan for."}
        </p>
      )}
    </div>
  );
};
