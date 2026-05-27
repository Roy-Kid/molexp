/**
 * Render a workflow IR dict as a runnable Python script.
 *
 * Mirrors :meth:`molexp.workflow.WorkflowCompiler.ir_to_python` so the
 * UI can preview the Python form of an in-memory IR — including edits
 * the user just made — without a server round-trip.
 *
 * The script assigns the IR to a top-level ``WORKFLOW_IR`` literal and
 * loads it via ``WorkflowSpec.from_dict``; the Python compiler does the
 * same so a user's edited Python can round-trip back through the
 * server.
 */

const SCRIPT_HEADER =
  '"""Auto-generated molexp workflow plan.\n\n' +
  "The IR literal below is the source of truth — edit it, and the\n" +
  "matching ``WorkflowSpec.from_dict(WORKFLOW_IR)`` call below will\n" +
  "load your edits. The server reads the IR (not this script) when\n" +
  "running the workflow; the script exists so the IR is comfortable\n" +
  'to read and review in Python form.\n"""\n' +
  "from molexp.workflow.spec import WorkflowSpec\n";

export const renderPythonFromIr = (ir: unknown): string => {
  if (!isPlainObject(ir)) {
    throw new Error("renderPythonFromIr: ir must be an object.");
  }
  const literal = pyRepr(ir, 0);
  return (
    `${SCRIPT_HEADER}\n` +
    `WORKFLOW_IR = ${literal}\n\n` +
    "spec = WorkflowSpec.from_dict(WORKFLOW_IR)\n"
  );
};

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

const pyString = (s: string): string => {
  // Use single quotes when possible to match Python's default repr.
  if (!s.includes("'") || s.includes('"')) {
    return `'${s.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, "\\n")}'`;
  }
  return `"${s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n")}"`;
};

const pad = (level: number): string => "    ".repeat(level);

const pyRepr = (value: unknown, level: number): string => {
  if (value === null || value === undefined) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error("renderPythonFromIr: non-finite numbers are not literal-safe.");
    }
    return String(value);
  }
  if (typeof value === "string") return pyString(value);
  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    const inner = value.map((v) => `${pad(level + 1)}${pyRepr(v, level + 1)}`).join(",\n");
    return `[\n${inner},\n${pad(level)}]`;
  }
  if (isPlainObject(value)) {
    const entries = Object.entries(value);
    if (entries.length === 0) return "{}";
    const inner = entries
      .map(([k, v]) => `${pad(level + 1)}${pyString(k)}: ${pyRepr(v, level + 1)}`)
      .join(",\n");
    return `{\n${inner},\n${pad(level)}}`;
  }
  throw new Error(`renderPythonFromIr: value of type ${typeof value} is not literal-safe.`);
};
