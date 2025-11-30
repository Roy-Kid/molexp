import { type Node, type Edge } from '@xyflow/react';
import type { TaskNodeJson, EdgeJson } from '@/types/task_graph_ir';

/**
 * Converts TaskGraph JSON IR to React Flow elements with auto-layout.
 * Uses a simple layer-based layout algorithm (Top-Down).
 */
export const getLayoutedElements = (nodes: TaskNodeJson[], edges: EdgeJson[]) => {
    const rankSep = 150; // Vertical separation between levels
    const nodeSep = 200; // Horizontal separation between nodes in same level

    // Build adjacency list and calculate in-degrees
    const adj: Record<string, string[]> = {};
    const inDegree: Record<string, number> = {};

    nodes.forEach(n => {
        adj[n.id] = [];
        inDegree[n.id] = 0;
    });

    edges.forEach(e => {
        if (adj[e.from]) adj[e.from].push(e.to);
        inDegree[e.to] = (inDegree[e.to] || 0) + 1;
    });

    // Calculate levels (longest path from source)
    const levels: Record<string, number> = {};
    const queue: string[] = nodes.filter(n => inDegree[n.id] === 0).map(n => n.id);

    queue.forEach(id => { levels[id] = 0; });

    // Topological sort / Level assignment
    const sortedNodes: string[] = [];
    const tempInDegree = { ...inDegree };

    while (queue.length > 0) {
        const u = queue.shift()!;
        sortedNodes.push(u);

        if (adj[u]) {
            adj[u].forEach(v => {
                levels[v] = Math.max(levels[v] || 0, (levels[u] || 0) + 1);
                tempInDegree[v]--;
                if (tempInDegree[v] === 0) {
                    queue.push(v);
                }
            });
        }
    }

    // Handle cycles or disconnected components not reached
    nodes.forEach(n => {
        if (levels[n.id] === undefined) levels[n.id] = 0;
    });

    // Group by level
    const levelGroups: Record<number, string[]> = {};
    let maxLevel = 0;
    Object.entries(levels).forEach(([id, level]) => {
        if (!levelGroups[level]) levelGroups[level] = [];
        levelGroups[level].push(id);
        maxLevel = Math.max(maxLevel, level);
    });

    // Calculate grid dimensions
    const maxNodesInLevel = Math.max(...Object.values(levelGroups).map(g => g.length));
    const totalWidth = maxNodesInLevel * nodeSep;

    // Create React Flow nodes
    const flowNodes: Node[] = nodes.map(node => {
        const level = levels[node.id];
        const indexInLevel = levelGroups[level].indexOf(node.id);
        const nodesInThisLevel = levelGroups[level].length;

        // Calculate position
        // Center the level horizontally
        const levelWidth = nodesInThisLevel * nodeSep;
        const xOffset = (totalWidth - levelWidth) / 2;

        const x = xOffset + indexInLevel * nodeSep + 50;
        const y = level * rankSep + 50;

        // Map type to React Flow type
        let type = 'process';
        const lowerType = node.type.toLowerCase();
        if (lowerType.includes('start') || lowerType.includes('load')) type = 'start';
        else if (lowerType.includes('end') || lowerType.includes('save')) type = 'end';

        return {
            id: node.id,
            type,
            position: { x, y },
            data: {
                label: node.label || node.type,
                ...node.params
            },
        };
    });

    const flowEdges: Edge[] = edges.map(edge => ({
        id: `e_${edge.from}_${edge.to}`,
        source: edge.from,
        target: edge.to,
        animated: true,
        type: 'smoothstep', // Better for top-down
    }));

    return { nodes: flowNodes, edges: flowEdges };
};
