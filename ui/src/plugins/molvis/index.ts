import { registerFileTypeContribution } from "@/app/registry";
import type { UiPluginModule } from "@/plugins/types";
import { MolvisTab } from "./MolvisTab";

const molvisPlugin: UiPluginModule = {
  id: "molvis",
  register: () => {
    registerFileTypeContribution({
      id: "molvis:run-tab",
      objectType: "run",
      value: "molvis",
      label: "LAMMPS",
      priority: 40,
      matcher: {
        // Logs (molpy.io.LAMMPSLog) + molvis-core supported formats:
        // pdb, xyz, lammps (data), lammps-dump (.dump/.lammpstrj/.lmptrj/.lammpsdump).
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
        ],
      },
      Component: MolvisTab,
    });
  },
};

export default molvisPlugin;
