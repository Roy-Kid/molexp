/**
 * molplot plugin tab — data-driven when MolRec / plot artifacts are present.
 *
 * Full MolRec observables → chart wiring lands with molrs browser readers;
 * for now we list matched files and render Vega specs when the payload is
 * already a Vega-Lite JSON document.
 */

import type { VegaLiteSpec } from "@molcrafts/molplot";
import type { JSX } from "react";
import { useEffect, useState } from "react";
import type { RendererProps } from "@/app/types";
import type { DiscoveredFile } from "@/plugins/types";
import { MolplotRawChart } from "./MolplotRawChart";

export const MolplotObservablesTab = ({
  discoveredFiles,
}: RendererProps & { discoveredFiles: DiscoveredFile[] }): JSX.Element => {
  const [selected, setSelected] = useState(discoveredFiles[0]?.relPath ?? "");
  const [spec, setSpec] = useState<VegaLiteSpec | null>(null);
  const [error, setError] = useState<string | null>(null);

  const file = discoveredFiles.find((f) => f.relPath === selected) ?? discoveredFiles[0];

  useEffect(() => {
    if (!file) return;
    setSpec(null);
    setError(null);
    // Only attempt JSON parse for explicit plot specs; MolRec dirs need molrs.
    if (!file.name.endsWith(".json") && !file.name.endsWith(".vl.json")) {
      setError(
        "MolRec observables need a molrs-backed reader in this tab (coming next). " +
          "For now land a Vega-Lite .vl.json or open Science products in Outputs.",
      );
      return;
    }
  }, [file]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-none gap-1 overflow-x-auto border-b border-border px-2 py-2">
        {discoveredFiles.map((f) => (
          <button
            key={f.relPath}
            type="button"
            onClick={() => setSelected(f.relPath)}
            className={
              (file?.relPath === f.relPath
                ? "bg-interactive text-foreground "
                : "text-muted-foreground hover:bg-interactive/60 ") +
              "rounded-[var(--radius-control)] px-2 py-1 font-mono text-micro"
            }
          >
            {f.name}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3">
        {error && <p className="text-sm text-muted-foreground">{error}</p>}
        {spec && <MolplotRawChart spec={{ spec }} style={{ width: "100%", height: 360 }} />}
        {!error && !spec && file && (
          <p className="text-sm text-muted-foreground">
            Matched plot product <span className="font-mono">{file.relPath}</span>. Plugins activate
            from MolRec <code className="text-micro">observables/</code> once the record is landed
            under the run.
          </p>
        )}
      </div>
    </div>
  );
};
