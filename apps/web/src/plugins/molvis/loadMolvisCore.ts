export const loadMolvisCore = async (): Promise<{
  mountMolvis: typeof import("@molcrafts/molvis-core")["mountMolvis"];
  loadFileContent: typeof import("@molcrafts/molvis-core/io")["loadFileContent"];
}> => {
  const [{ mountMolvis }, { loadFileContent }] = await Promise.all([
    import("@molcrafts/molvis-core"),
    import("@molcrafts/molvis-core/io"),
  ]);
  return { mountMolvis, loadFileContent };
};
