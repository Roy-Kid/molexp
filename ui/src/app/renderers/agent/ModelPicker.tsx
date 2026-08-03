/**
 * Inline model selector for the agent composer.
 *
 * Lists models already configured on the operator (global tiers + active
 * model). Changing selection updates the default model via PUT /api/agent/provider
 * — no trip to the Settings page.
 */

import { ChevronDown, Cpu, Loader2 } from "lucide-react";
import { type JSX, useCallback, useEffect, useMemo, useState } from "react";

import { type ApiAgentProvider, type ApiTierModels, agentAdminApi } from "@/app/state/api";
import { WorkbenchAction, WorkbenchOperationState } from "@/components/workbench";
import { cn } from "@/lib/utils";

const TIER_KEYS = ["default", "cheap", "heavy"] as const;

/** Prefer the short model id after `provider:` for compact UI labels. */
export const modelDisplayName = (qualified: string): string => {
  const i = qualified.indexOf(":");
  return i >= 0 ? qualified.slice(i + 1) : qualified;
};

/**
 * Collect unique non-empty model ids from a provider response.
 *
 * Operator config often carries the same logical model twice — once bare
 * (``deepseek-v4-flash``) and once qualified (``deepseek:deepseek-v4-flash``),
 * or the same id on both ``model`` and ``models.default``. Dedupe by the
 * display name (strip ``provider:``); prefer the qualified form for the
 * stored value so the select options are not shown twice.
 */
export const collectConfiguredModels = (provider: ApiAgentProvider): string[] => {
  const byDisplay = new Map<string, string>();
  const add = (value: string | undefined | null): void => {
    const v = (value ?? "").trim();
    if (!v) return;
    const display = modelDisplayName(v).toLowerCase();
    const existing = byDisplay.get(display);
    // Prefer provider:model over bare ids when both appear.
    if (!existing || (v.includes(":") && !existing.includes(":"))) {
      byDisplay.set(display, v);
    }
  };
  add(provider.model);
  for (const key of TIER_KEYS) {
    add(provider.models?.[key]);
  }
  for (const cfg of provider.configurations ?? []) {
    for (const key of TIER_KEYS) {
      add(cfg.models?.[key]);
    }
  }
  return Array.from(byDisplay.values());
};

export interface ModelPickerProps {
  className?: string;
  disabled?: boolean;
  disabledReason?: string;
  /** Called after a successful model switch (optional parent refresh). */
  onChanged?: (provider: ApiAgentProvider) => void;
}

export const ModelPicker = ({
  className,
  disabled = false,
  disabledReason = "Model switching is unavailable while the agent is responding.",
  onChanged,
}: ModelPickerProps): JSX.Element | null => {
  const [options, setOptions] = useState<string[]>([]);
  const [active, setActive] = useState("");
  const [tiers, setTiers] = useState<ApiTierModels | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [providerError, setProviderError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [lastRequested, setLastRequested] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const hydrate = useCallback(async (): Promise<void> => {
    setLoading(true);
    setProviderError(null);
    try {
      const p = await agentAdminApi.getProvider();
      setOptions(collectConfiguredModels(p));
      setActive(p.model || p.models?.default || "");
      setTiers(p.models ?? null);
    } catch (err) {
      setOptions([]);
      setActive("");
      setTiers(null);
      setProviderError(err instanceof Error ? err.message : "Failed to load configured models.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void hydrate();
  }, [hydrate]);

  useEffect(() => {
    if (!success) return;
    const handle = window.setTimeout(() => setSuccess(null), 3000);
    return () => window.clearTimeout(handle);
  }, [success]);

  const sortedOptions = useMemo(() => {
    const list = [...options];
    if (active && !list.includes(active)) list.unshift(active);
    return list.sort((a, b) => modelDisplayName(a).localeCompare(modelDisplayName(b)));
  }, [options, active]);

  const handleChange = async (next: string): Promise<void> => {
    if (!next || next === active || saving || disabled) return;
    setSaving(true);
    setLastRequested(next);
    setSaveError(null);
    setSuccess(null);
    try {
      // Keep cheap/heavy; point default + legacy `model` at the selection so
      // chat and plan routers both pick it up on the next turn.
      const models: ApiTierModels = {
        cheap: tiers?.cheap || next,
        default: next,
        heavy: tiers?.heavy || next,
      };
      const updated = await agentAdminApi.updateProvider({ model: next, models });
      setActive(updated.model || next);
      setTiers(updated.models ?? models);
      setOptions(collectConfiguredModels(updated));
      setSuccess(`Model switched to ${modelDisplayName(updated.model || next)}.`);
      onChanged?.(updated);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Failed to switch model.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <WorkbenchOperationState
        kind="loading"
        density="inline"
        title="Loading models…"
        className={cn("px-2 py-1", className)}
      />
    );
  }

  if (providerError) {
    return (
      <WorkbenchOperationState
        kind="error"
        density="inline"
        title="Models unavailable"
        detail={providerError}
        className={className}
        action={
          <WorkbenchAction
            kind="ghost"
            size="compact"
            className="h-6 px-2 text-micro"
            onClick={() => void hydrate()}
          >
            Retry
          </WorkbenchAction>
        }
      />
    );
  }

  if (sortedOptions.length === 0) {
    return (
      <WorkbenchOperationState
        kind="disabled"
        density="inline"
        title="No models configured"
        detail="Configure an agent provider in Settings."
        className={className}
      />
    );
  }

  return (
    <span className="inline-flex flex-col items-start gap-1">
      <label
        className={cn(
          "inline-flex max-w-[14rem] items-center gap-1 rounded-md px-2 py-1 font-mono text-micro",
          "text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
          (disabled || saving) && "opacity-60",
          className,
        )}
        title={disabled ? disabledReason : "Active model — pick another configured model"}
        aria-busy={saving}
      >
        <Cpu className="h-3 w-3 flex-none opacity-60" aria-hidden />
        <span className="relative inline-flex min-w-0 items-center">
          <select
            className={cn(
              "max-w-[11rem] cursor-pointer appearance-none truncate bg-transparent pr-4",
              "border-0 py-0 text-micro font-mono text-inherit outline-none",
              "focus-visible:ring-1 focus-visible:ring-ring rounded-sm",
            )}
            value={active}
            disabled={disabled || saving}
            aria-label="Select agent model"
            aria-description={disabled ? disabledReason : undefined}
            onChange={(e) => {
              void handleChange(e.target.value);
            }}
          >
            {sortedOptions.map((id) => (
              <option key={id} value={id} title={id}>
                {modelDisplayName(id)}
              </option>
            ))}
          </select>
          {saving ? (
            <Loader2
              className="mol-motion-progress-spin pointer-events-none absolute right-0 h-3 w-3 text-status-running"
              aria-hidden
            />
          ) : (
            <ChevronDown className="pointer-events-none absolute right-0 h-3 w-3 opacity-50" />
          )}
        </span>
      </label>
      {saving && (
        <WorkbenchOperationState
          kind="running"
          density="inline"
          title="Switching model…"
          className="sr-only"
        />
      )}
      {saveError && (
        <WorkbenchOperationState
          kind="error"
          density="inline"
          title="Model switch failed"
          detail={saveError}
          action={
            lastRequested ? (
              <WorkbenchAction
                kind="ghost"
                size="compact"
                className="h-6 px-2 text-micro"
                onClick={() => void handleChange(lastRequested)}
              >
                Retry
              </WorkbenchAction>
            ) : undefined
          }
        />
      )}
      {success && <WorkbenchOperationState kind="success" density="inline" title={success} />}
    </span>
  );
};
