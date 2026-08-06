import { registerFileTypeContribution } from "@/app/registry";
import { filePreviewPluginRegistry } from "@/lib/file-preview-plugins";
import type { UiPluginModule } from "@/plugins/types";
import { MolvisDatasetPreview } from "./MolvisDatasetPreview";
import { MolvisTab } from "./MolvisTab";

/**
 * Classic trajectory / log formats molvis can render in-browser today.
 * MolRec Zarr ``frame/`` / ``trajectory/`` (molrs) is not opened here yet —
 * see molrec storage L4; tab still surfaces when a Zarr record root is present
 * so users see the package in the file rail.
 */
const isClassicMolvisFile = (file: { name: string; relPath: string }): boolean => {
  const name = file.name.toLowerCase();
  return (
    name === "log.lammps" ||
    name.endsWith(".lammps.log") ||
    name === "lmp.log" ||
    /\.(lammpstrj|lmptrj|lammpsdump|dump|xyz|extxyz|pdb)$/i.test(name)
  );
};

/** Zarr V3 group metadata under a MolRec-ish package (frame/trajectory/meta). */
const isMolrecZarrMarker = (file: { name: string; relPath: string }): boolean => {
  const path = file.relPath.toLowerCase().replace(/\\/g, "/");
  if (file.name.toLowerCase() !== "zarr.json") return false;
  return (
    path.includes("/frame/") ||
    path.includes("/trajectory/") ||
    path.endsWith("/meta/zarr.json") ||
    path.endsWith("/frame/zarr.json") ||
    path.endsWith("/trajectory/zarr.json")
  );
};

const molvisPlugin: UiPluginModule = {
  id: "molvis",
  register: () => {
    registerFileTypeContribution({
      id: "molvis:run-tab",
      objectType: "run",
      value: "molvis",
      label: "MolVis",
      priority: 40,
      matcher: {
        // Logs (molpy.io.LAMMPSLog) + molvis-core supported formats +
        // MolRec Zarr markers (discovery only until a Zarr frame reader ships).
        patterns: [
          "log.lammps",
          "**/log.lammps",
          "*.lammps.log",
          "**/*.lammps.log",
          "lmp.log",
          "**/lmp.log",
          "*.lammpstrj",
          "**/*.lammpstrj",
          "*.lmptrj",
          "**/*.lmptrj",
          "*.lammpsdump",
          "**/*.lammpsdump",
          "*.dump",
          "**/*.dump",
          "*.xyz",
          "**/*.xyz",
          "*.extxyz",
          "**/*.extxyz",
          "*.pdb",
          "**/*.pdb",
          "**/meta/zarr.json",
          "**/frame/zarr.json",
          "**/trajectory/zarr.json",
          "**/frame/**/zarr.json",
          "**/trajectory/**/zarr.json",
        ],
        // Sidecar-backed datasets match no extension — the server flags
        // them via a same-stem `.py` reader sidecar. See
        // molexp.server.preview and GET /api/assets/{id}/preview.
        matches: (file) =>
          file.hasPreviewSidecar === true ||
          isClassicMolvisFile(file) ||
          isMolrecZarrMarker(file),
      },
      Component: MolvisTab,
    });

    filePreviewPluginRegistry.register({
      id: "molvis:dataset-preview",
      name: "Molvis",
      extensions: [],
      canHandle: ({ hasPreviewSidecar }) => hasPreviewSidecar === true,
      Component: MolvisDatasetPreview,
    });
  },
};

export default molvisPlugin;
