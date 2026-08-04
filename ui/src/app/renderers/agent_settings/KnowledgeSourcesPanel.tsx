/**
 * Knowledge package pin for the **molmcp** MCP server (per-MCP, not global).
 *
 * Writes ``env.MOLMCP_SOURCES`` on that server's config row. Plan sessions
 * and capability grounding read the same pin from the MCP store.
 */

import { BookOpen, Check, Server, X } from "lucide-react";
import type { JSX } from "react";
import { useCallback, useEffect, useState } from "react";
import { AgentUnavailableError, resetAgentProbes } from "@/app/state/agentProbe";
import { type ApiKnowledgeSources, agentAdminApi } from "@/app/state/api";
import { Code as InlineCode } from "@/components/ui/code";
import { ProgressSpinner } from "@/components/ui/progress-spinner";
import { WorkbenchAction, WorkbenchIconAction, WorkbenchTag } from "@/components/workbench";
import { cn } from "@/lib/utils";
import { UnavailableCapability } from "./UnavailableCapability";

export const KnowledgeSourcesPanel = (): JSX.Element => {
  const [data, setData] = useState<ApiKnowledgeSources | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    setUnavailable(false);
    try {
      const res = await agentAdminApi.getKnowledgeSources();
      setData(res);
      setSelected([...res.sources]);
    } catch (err) {
      if (err instanceof AgentUnavailableError) setUnavailable(true);
      else setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const toggle = (pkg: string): void => {
    setSelected((prev) => (prev.includes(pkg) ? prev.filter((p) => p !== pkg) : [...prev, pkg]));
  };

  const save = async (): Promise<void> => {
    setSaving(true);
    setError(null);
    try {
      const res = await agentAdminApi.updateKnowledgeSources(selected);
      setData(res);
      setSelected([...res.sources]);
      setSavedFlash(true);
      window.setTimeout(() => setSavedFlash(false), 1500);
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  };

  if (unavailable) {
    return (
      <UnavailableCapability
        title="MCP knowledge settings unavailable"
        description="Could not reach agent admin APIs for molmcp package scope."
        onRetry={() => {
          resetAgentProbes();
          void refresh();
        }}
      />
    );
  }

  const packages = data?.knownPackages ?? [
    "molpy",
    "molpack",
    "molvis",
    "molplot",
    "molq",
    "molcfg",
    "atomiverse",
    "lammps",
  ];
  const dirty =
    JSON.stringify([...selected].sort()) !== JSON.stringify([...(data?.sources ?? [])].sort());

  return (
    <section className="space-y-3 bg-surface/60 px-3 py-3">
      <header className="flex items-start gap-2">
        <BookOpen className="mt-0.5 size-4 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <h3 className="text-body-lg font-semibold text-foreground">
            molmcp · knowledge packages
          </h3>
          <p className="text-label text-muted-foreground">
            Per-MCP scope for the <InlineCode className="text-micro">molmcp</InlineCode> server
            (stored as <InlineCode className="text-micro">MOLMCP_SOURCES</InlineCode>). Plan
            sessions and tool calls inherit this pin — e.g. only molpy + molvis + molplot, never
            atomiverse.
          </p>
        </div>
        {data?.configured ? (
          data.unrestricted ? (
            <WorkbenchTag meaning="metadata" className="text-label">
              all packages
            </WorkbenchTag>
          ) : (
            <WorkbenchTag className="text-label">{selected.length} pinned</WorkbenchTag>
          )
        ) : (
          <WorkbenchTag meaning="failed" className="text-label">
            molmcp missing
          </WorkbenchTag>
        )}
      </header>

      {data && (
        <p className="flex items-center gap-1.5 text-micro text-muted-foreground">
          <Server className="size-3" />
          <span className="font-mono">{data.serverName}</span>
          <span>· {data.scope} scope</span>
        </p>
      )}

      {!loading && data && !data.configured ? (
        <p className="text-label text-warning-foreground">
          Add an MCP server named <InlineCode className="text-micro">molmcp</InlineCode> below
          first, then set package scope here.
        </p>
      ) : null}

      {loading ? (
        <p className="text-label text-muted-foreground">Loading…</p>
      ) : (
        <div className="flex flex-wrap gap-2">
          {packages.map((pkg) => {
            const on = selected.includes(pkg);
            return (
              <WorkbenchAction
                kind="ghost"
                size="content"
                key={pkg}
                type="button"
                disabled={!data?.configured}
                onClick={() => toggle(pkg)}
                className={cn(
                  "rounded-control px-2.5 py-1 font-mono text-label transition-colors disabled:opacity-40",
                  on
                    ? "bg-info-soft/60 text-info-foreground"
                    : "bg-muted/40 text-muted-foreground hover:bg-muted",
                )}
              >
                {on ? "✓ " : ""}
                {pkg}
              </WorkbenchAction>
            );
          })}
        </div>
      )}

      {error ? <p className="text-label text-destructive">{error}</p> : null}

      <div className="flex items-center justify-end gap-2">
        <WorkbenchIconAction
          label="Clear all knowledge packages"
          disabled={loading || !data?.configured || selected.length === 0}
          onClick={() => setSelected([])}
        >
          <X className="size-4" />
        </WorkbenchIconAction>
        <WorkbenchAction
          kind="primary"
          size="compact"
          disabled={loading || saving || !dirty || !data?.configured}
          onClick={() => void save()}
        >
          {saving ? (
            <ProgressSpinner className="mr-1" label="Saving" />
          ) : savedFlash ? (
            <Check className="mr-1 size-3.5" />
          ) : null}
          {savedFlash ? "Saved" : "Save on molmcp"}
        </WorkbenchAction>
      </div>
    </section>
  );
};
