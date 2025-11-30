/**
 * TaskGraphCompiler for TypeScript/React frontend.
 * 
 * Provides bidirectional conversion between:
 * - UI graph state (ReactFlow nodes and edges)
 * - JSON intermediate representation (TaskGraphJson)
 * 
 * This compiler is the TypeScript counterpart to the Python TaskGraphCompiler.
 * Both must maintain identical field names and structure for the JSON IR.
 */

import { type Node, type Edge } from '@xyflow/react';
import {
    type TaskGraphJson,
    type TaskNodeJson,
    type EdgeJson,
} from '@/types/task_graph_ir';

/**
 * Compiler for converting between UI graph state and JSON IR.
 * 
 * Responsibilities:
 * - Convert ReactFlow nodes/edges to JSON IR (uiToJson)
 * - Convert JSON IR to ReactFlow nodes/edges (jsonToUi)
 * - Preserve layout information in metadata
 * - Handle missing/default values gracefully
 * 
 * The compiler performs pure transformations without side effects.
 */
export class TaskGraphCompiler {
    /**
     * Convert UI graph state to JSON intermediate representation.
     * 
     * Takes the current state of the graph editor (ReactFlow nodes and edges)
     * and produces a JSON IR that can be sent to the backend.
     * 
     * @param name - Workflow name
     * @param nodes - ReactFlow nodes from the graph editor
     * @param edges - ReactFlow edges from the graph editor
     * @param metadata - Optional workflow-level metadata
     * @returns JSON IR conforming to TaskGraphJson schema
     */
    static uiToJson(
        name: string,
        nodes: Node[],
        edges: Edge[],
        metadata?: Record<string, any>
    ): TaskGraphJson {
        // Convert UI nodes to JSON IR nodes
        const taskNodes: TaskNodeJson[] = nodes.map((node) => ({
            id: node.id,
            type: node.type || 'unknown',
            label: (node.data?.label as string) || null,
            params: node.data?.config || {},
            metadata: {
                // Preserve position for layout
                position: node.position,
                // Include any additional UI metadata
                ...(node.data?.metadata || {}),
            },
        }));

        // Convert UI edges to JSON IR edges
        const taskEdges: EdgeJson[] = edges.map((edge) => ({
            from: edge.source,
            to: edge.target,
            kind: edge.type || 'depends',
            metadata: edge.data || {},
        }));

        return {
            name,
            nodes: taskNodes,
            edges: taskEdges,
            version: metadata?.version || null,
            metadata,
        };
    }

    /**
     * Convert JSON intermediate representation to UI graph state.
     * 
     * Takes JSON IR (e.g., from backend export) and produces ReactFlow
     * nodes and edges that can be loaded into the graph editor.
     * 
     * @param graph - JSON IR conforming to TaskGraphJson schema
     * @returns Object with ReactFlow-compatible nodes and edges
     */
    static jsonToUi(graph: TaskGraphJson): { nodes: Node[]; edges: Edge[] } {
        // Convert JSON IR nodes to UI nodes
        const nodes: Node[] = graph.nodes.map((node) => {
            // Extract position from metadata, or use default
            const position = node.metadata?.position || { x: 0, y: 0 };

            return {
                id: node.id,
                type: node.type,
                position,
                data: {
                    label: node.label || node.type,
                    config: node.params,
                    metadata: node.metadata,
                },
            };
        });

        // Convert JSON IR edges to UI edges
        const edges: Edge[] = graph.edges.map((edge, idx) => {
            // Generate unique edge ID
            const edgeId = `e-${edge.from}-${edge.to}-${idx}`;

            return {
                id: edgeId,
                source: edge.from,
                target: edge.to,
                type: edge.kind || 'default',
                data: edge.metadata,
            };
        });

        return { nodes, edges };
    }

    /**
     * Validate JSON IR structure (basic validation).
     * 
     * Checks for required fields and basic type correctness.
     * For full validation, use the backend /api/workflows/validate endpoint.
     * 
     * @param payload - Object to validate
     * @returns Validation result with error message if invalid
     */
    static validate(payload: any): { valid: boolean; error?: string } {
        if (!payload || typeof payload !== 'object') {
            return { valid: false, error: 'Payload must be an object' };
        }

        if (!payload.name || typeof payload.name !== 'string') {
            return { valid: false, error: 'Missing or invalid field: name' };
        }

        if (!Array.isArray(payload.nodes)) {
            return { valid: false, error: 'Field "nodes" must be an array' };
        }

        if (!Array.isArray(payload.edges)) {
            return { valid: false, error: 'Field "edges" must be an array' };
        }

        // Validate each node has required fields
        for (let i = 0; i < payload.nodes.length; i++) {
            const node = payload.nodes[i];
            if (!node.id) {
                return { valid: false, error: `Node ${i} missing required field: id` };
            }
            if (!node.type) {
                return {
                    valid: false,
                    error: `Node ${i} missing required field: type`,
                };
            }
        }

        // Validate each edge has required fields
        for (let i = 0; i < payload.edges.length; i++) {
            const edge = payload.edges[i];
            if (!edge.from) {
                return {
                    valid: false,
                    error: `Edge ${i} missing required field: from`,
                };
            }
            if (!edge.to) {
                return { valid: false, error: `Edge ${i} missing required field: to` };
            }
        }

        return { valid: true };
    }
}
