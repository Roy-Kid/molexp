export const loadMolvisCore = async (): Promise<{
  mountMolvis: typeof import("@molcrafts/molvis-stage")["mountMolvis"];
  loadFileContent: typeof import("@molcrafts/molvis-stage/io")["loadFileContent"];
}> => {
  const [{ mountMolvis }, { loadFileContent }] = await Promise.all([
    import("@molcrafts/molvis-stage"),
    import("@molcrafts/molvis-stage/io"),
  ]);
  return { mountMolvis, loadFileContent };
};
