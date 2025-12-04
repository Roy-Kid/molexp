/**
 * JSON Intermediate Representation (IR) for task graphs.
 * 
 * These types define the language-agnostic JSON schema shared between
 * the Python backend and TypeScript frontend. The field names and structure
 * must match exactly on both sides.
 * 
 * Naming convention: snake_case (consistent with Python)
 */

/**
 * Top-level task graph structure.
 * 
 * Represents a complete workflow with nodes (tasks) and edges (dependencies).
 */
export interface TaskGraphJson {
    /** Human-readable workflow name */
    name: string;

    /** Optional version identifier */
    version?: string | null;

    /** List of task nodes in the graph */
    nodes: TaskNodeJson[];

    /** List of directed edges between nodes */
    edges: EdgeJson[];

    /** Additional workflow-level metadata */
    metadata?: Record<string, any>;

    /** List of target node IDs to execute */
    targets?: string[];
}

/**
 * Individual task node in the graph.
 * 
 * Each node represents a unit of work (task) with its configuration.
 */
export interface TaskNodeJson {
    /** Unique identifier within this graph */
    id: string;

    /** Task type/class name (e.g., "LoadMolecule", "OptimizeGeometry") */
    type: string;

    /** Human-readable label for display */
    label?: string | null;

    /** Task-specific configuration parameters */
    params: Record<string, any>;

    /** Additional metadata (e.g., UI layout, tags) */
    metadata?: Record<string, any>;
}

/**
 * Directed edge between two nodes.
 * 
 * Represents a dependency relationship in the task graph.
 */
export interface EdgeJson {
    /** Source node ID */
    from: string;

    /** Target node ID */
    to: string;

    /** Edge type (e.g., "depends", "data"). Defaults to "depends" */
    kind?: string | null;

    /** Additional edge metadata */
    metadata?: Record<string, any>;
}
