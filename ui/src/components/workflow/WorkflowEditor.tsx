import React, { useCallback, useRef, useState, useEffect } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type Edge,
  ReactFlowProvider,
  type Node,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Play, Plus, Layout, Maximize, ZoomIn, ZoomOut, Save } from 'lucide-react';
import { Panel, useReactFlow } from '@xyflow/react';
import { Separator } from '../ui/separator';
import { useNavigate } from 'react-router-dom';

import { ProcessNode } from './nodes/ProcessNode';
import { NodePaletteDialog } from './NodePaletteDialog';
import { NodeDetailPanel } from './NodeDetailPanel';
import { NodeConfigDialog } from './NodeConfigDialog';
import { ContextMenu } from './ContextMenu';
import { Button } from '../ui/button';
import { useAppStore } from '@/store/useAppStore';
import { getLayoutedElements, autoLayoutNodes, toTaskGraphJson, planExecution, analyzePaths } from '@/lib/workflow-utils';
import type { TaskGraphJson } from '@/types/task_graph_ir';

const PATH_COLORS = [
  '#2563eb', // Blue
  '#9333ea', // Purple
  '#db2777', // Pink
  '#ea580c', // Orange
  '#059669', // Emerald
  '#0891b2', // Cyan
];

const SHARED_COLOR = '#475569'; // Slate-600

const nodeTypes = {
  process: ProcessNode,
  start: ProcessNode, // Fallback for legacy start nodes
  end: ProcessNode,   // Fallback for legacy end nodes
  // Built-in nodes
  'io.write_file': ProcessNode,
  'data.read_json': ProcessNode,
  'data.write_json': ProcessNode,
  'http.request': ProcessNode,
  'text.transform': ProcessNode,
  'debug.inspect': ProcessNode,
  // Legacy/Example nodes
  'optimize-geometry': ProcessNode,
  'calc-energy': ProcessNode,
  'run-md': ProcessNode,
};

const initialNodes: Node[] = [];




interface WorkflowEditorProps {
  readOnly?: boolean;
  initialNodes?: Node[];
  initialEdges?: Edge[];
  initialGraph?: TaskGraphJson;
  executionStatus?: Record<string, 'success' | 'failed' | 'pending' | 'running'>;
  onNodeSelect?: (node: Node | null) => void;
  onSave?: (graph: TaskGraphJson) => Promise<void>;
  onChange?: () => void;
}

export const WorkflowEditor = ({ 
  readOnly = false, 
  initialNodes: propNodes, 
  initialEdges: propEdges,
  initialGraph,
  executionStatus,
  onNodeSelect,
  onSave,
  onChange
}: WorkflowEditorProps) => {
  return (
    <ReactFlowProvider>
      <WorkflowEditorContent 
        readOnly={readOnly} 
        initialNodes={propNodes} 
        initialEdges={propEdges}
        initialGraph={initialGraph}
        executionStatus={executionStatus}
        onNodeSelect={onNodeSelect}
        onSave={onSave}
        onChange={onChange}
      />
    </ReactFlowProvider>
  );
};

const WorkflowEditorContent = ({ 
  readOnly, 
  initialNodes: propNodes, 
  initialEdges: propEdges,
  initialGraph,
  executionStatus,
  onNodeSelect,
  onSave,
  onChange
}: WorkflowEditorProps) => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  
  // Initialize state based on props
  const getInitialState = () => {
    if (initialGraph) {
      const { nodes, edges } = getLayoutedElements(initialGraph.nodes, initialGraph.edges);
      return { nodes, edges };
    }
    return { 
      nodes: propNodes || initialNodes, 
      edges: propEdges || [] 
    };
  };

  const initialState = getInitialState();

  const [nodes, setNodes, onNodesChange] = useNodesState(initialState.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialState.edges);
  const reactFlowInstance = useReactFlow();
  
  const [pathMap, setPathMap] = useState<Record<string, string[]>>({});
  const [targetColors, setTargetColors] = useState<Record<string, string>>({});

  // Update nodes and edges when executionStatus or pathMap changes
  useEffect(() => {
    setEdges((eds) => 
      eds.map(edge => {
        const status = executionStatus?.[edge.id];
        const isRunning = status === 'running';
        const isSuccess = status === 'success';
        const isFailed = status === 'failed';
        
        const paths = pathMap[edge.id] || [];
        const isPlanned = paths.length > 0;
        
        let stroke = '#94a3b8';
        if (isSuccess) stroke = '#16a34a';
        else if (isFailed) stroke = '#dc2626';
        else if (isRunning) stroke = '#3b82f6';
        else if (isPlanned) {
            stroke = paths.length === 1 ? targetColors[paths[0]] : SHARED_COLOR;
        }

        return {
          ...edge,
          style: {
            stroke,
            strokeWidth: isRunning || isPlanned ? 3 : 2,
            strokeDasharray: isPlanned ? '5,5' : undefined,
          },
          animated: isRunning || isPlanned,
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: stroke,
          },
        };
      })
    );

    setNodes((nds) => 
      nds.map(node => {
        const paths = pathMap[node.id] || [];
        const isPlanned = paths.length > 0;
        const plannedColor = isPlanned ? (paths.length === 1 ? targetColors[paths[0]] : SHARED_COLOR) : undefined;

        return {
          ...node,
          data: {
            ...node.data,
            status: executionStatus?.[node.id] || (isPlanned ? 'planned' : undefined),
            plannedColor // Pass the specific color to the node
          },
        };
      })
    );
  }, [executionStatus, pathMap, targetColors, setEdges, setNodes]);
  
  const [availableNodes, setAvailableNodes] = useState<any[]>([]);
  const [pendingNodeToAdd, setPendingNodeToAdd] = useState<string | null>(null);

  useEffect(() => {
    if (!readOnly) {
      fetch('/api/nodes')
        .then(res => res.json())
        .then(data => setAvailableNodes(data.nodes || []))
        .catch(err => console.error('Failed to fetch nodes:', err));
    }
  }, [readOnly]);

  const handleNodeAdd = (type: string) => {
    if (readOnly) return;
    setIsPaletteOpen(false);
    setPendingNodeToAdd(type);
    setEditingNodeData({}); // Start with empty config
    setIsConfigDialogOpen(true);
  };

  const handleNodeConfigUpdate = (data: any) => {
    if (readOnly) return;

    if (pendingNodeToAdd) {
      // Create new node
      const newNode: Node = {
        id: `node_${Date.now()}`, // Use timestamp to avoid collisions
        type: pendingNodeToAdd,
        position: { x: Math.random() * 400 + 100, y: Math.random() * 400 + 100 },
        data: { 
          label: pendingNodeToAdd.split('.').pop() || pendingNodeToAdd, 
          category: pendingNodeToAdd.split('.')[0],
          ...data 
        },
      };
      setNodes((nds) => nds.concat(newNode));
      setPendingNodeToAdd(null);
      onChange?.();
    } else if (editingNodeId) {
      // Update existing node
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === editingNodeId) {
            return { ...node, data: { ...node.data, ...data } };
          }
          return node;
        })
      );
      setEditingNodeId(null);
      onChange?.();
    }
    setIsConfigDialogOpen(false);
    setEditingNodeData(null);
  };

  const [isPaletteOpen, setIsPaletteOpen] = useState(false);
  const [isConfigDialogOpen, setIsConfigDialogOpen] = useState(false);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
  const [editingNodeData, setEditingNodeData] = useState<any>(null);

  const [menu, setMenu] = useState<{ id: string; top: number; left: number; type: 'node' | 'edge' } | null>(null);
  const addExecution = useAppStore((state) => state.addExecution);
  const navigate = useNavigate();

  const handleRunWorkflow = async () => {
    // Collect output nodes
    const targets = nodes.filter(n => n.data.isOutput).map(n => n.id);
    
    if (targets.length === 0) {
      alert("Please select at least one output node (Right click -> Set as Output)");
      return;
    }

    // Create execution with snapshot
    const snapshot = toTaskGraphJson(nodes, edges, `Workflow Run ${new Date().toLocaleTimeString()}`);
    
    // Ensure targets are set in snapshot (toTaskGraphJson handles this based on isOutput data)
    // But we double check here
    if (!snapshot.targets || snapshot.targets.length === 0) {
        // This should be caught by the check above, but just in case
        snapshot.targets = targets;
    }

    await addExecution(snapshot.name, snapshot);
    
    navigate('/executions');
  };

  const onConnect = useCallback(
    (params: Connection) => {
      if (readOnly) return;
      setEdges((eds) => addEdge({ 
        ...params, 
        type: 'default', // Ensure Bezier curve
        animated: false,
        style: { strokeWidth: 2, stroke: '#94a3b8' },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' }
      }, eds));
      onChange?.();
    },
    [setEdges, readOnly, onChange],
  );

  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      event.preventDefault();

      if (!reactFlowWrapper.current) return;
      const pane = reactFlowWrapper.current.getBoundingClientRect();
      
      setMenu({
        id: node.id,
        top: event.clientY - pane.top,
        left: event.clientX - pane.left,
        type: 'node',
      });
    },
    []
  );

  const onEdgeContextMenu = useCallback(
    (event: React.MouseEvent, edge: Edge) => {
      event.preventDefault();

      if (!reactFlowWrapper.current) return;
      const pane = reactFlowWrapper.current.getBoundingClientRect();

      setMenu({
        id: edge.id,
        top: event.clientY - pane.top,
        left: event.clientX - pane.left,
        type: 'edge',
      });
    },
    []
  );

  const onPaneClick = useCallback(() => {
    setMenu(null);
    setSelectedNode(null);
    if (onNodeSelect) onNodeSelect(null);
  }, [onNodeSelect]);

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
    if (onNodeSelect) onNodeSelect(node);
  }, [onNodeSelect]);

  const onNodeDoubleClick = useCallback((_event: React.MouseEvent, node: Node) => {
    if (readOnly) return;
    setEditingNodeId(node.id);
    setEditingNodeData(node.data);
    setIsConfigDialogOpen(true);
  }, [readOnly]);

  const handleEdit = () => {
    if (!menu || menu.type !== 'node') return;
    const node = nodes.find((n) => n.id === menu.id);
    if (node) {
      setEditingNodeId(node.id);
      setEditingNodeData(node.data);
      setIsConfigDialogOpen(true);
    }
    setMenu(null);
  };

  const handleDelete = () => {
    if (!menu) return;
    if (menu.type === 'node') {
        setNodes((nds) => nds.filter((node) => node.id !== menu.id));
        setEdges((eds) => eds.filter((edge) => edge.source !== menu.id && edge.target !== menu.id));
    } else if (menu.type === 'edge') {
        setEdges((eds) => eds.filter((edge) => edge.id !== menu.id));
    }
    setMenu(null);
    setSelectedNode(null);
    onChange?.();
  };

  const handleViewResults = () => {
    if (!menu || menu.type !== 'node') return;
    const node = nodes.find((n) => n.id === menu.id);
    if (node) {
      setSelectedNode(node);
      if (onNodeSelect) onNodeSelect(node);
    }
    setMenu(null);
  };

  const handleAutoLayout = useCallback(() => {
    const { nodes: layoutedNodes, edges: layoutedEdges } = autoLayoutNodes(nodes, edges);
    setNodes([...layoutedNodes]);
    setEdges([...layoutedEdges]);
  }, [nodes, edges, setNodes, setEdges]);

  const toggleOutput = useCallback((nodeId: string) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          const isOutput = !node.data.isOutput;
          return {
            ...node,
            data: { ...node.data, isOutput },
            // Style is now handled in ProcessNode based on data.isOutput
          };
        }
        return node;
      })
    );
    setMenu(null);
  }, [setNodes]);

  return (
    <div className="flex h-full w-full">
      {/* Main Workflow Area */}
      <div className="flex-1 h-full relative" ref={reactFlowWrapper}>
        <div className="absolute top-4 left-4 z-10 flex gap-2">
          {!readOnly && (
            <Button onClick={() => setIsPaletteOpen(true)} className="shadow-lg">
              <Plus className="mr-2 h-4 w-4" />
              Add Node
            </Button>
          )}
        </div>

        {!readOnly && (
          <div className="absolute top-4 right-4 z-10 flex gap-2">
            {onSave && (
              <Button 
                onClick={async () => {
                  const snapshot = toTaskGraphJson(nodes, edges, initialGraph?.name || 'Workflow');
                  await onSave(snapshot);
                }} 
                variant="secondary" 
                className="shadow-lg bg-white/90 hover:bg-white"
              >
                <Save className="mr-2 h-4 w-4" />
                Save
              </Button>
            )}
            <Button onClick={() => {
                // Dry Run Logic (Frontend Side)
                const targets = nodes.filter(n => n.data.isOutput).map(n => n.id);
                if (targets.length === 0) {
                    alert("Please select at least one output node.");
                    return;
                }
                
                try {
                    const analysis = analyzePaths(nodes, edges, targets);
                    
                    // Assign colors to targets
                    const newTargetColors: Record<string, string> = {};
                    targets.forEach((t, i) => {
                        newTargetColors[t] = PATH_COLORS[i % PATH_COLORS.length];
                    });
                    
                    setTargetColors(newTargetColors);
                    setPathMap(analysis);
                    
                    const totalNodes = Object.keys(analysis).filter(k => nodes.find(n => n.id === k)).length;
                    // alert(`Execution Plan:\n${totalNodes} nodes involved.`);
                } catch (err) {
                    alert(`Dry Run Failed: ${err}`);
                }
            }} variant="secondary" className="shadow-lg bg-white/90 hover:bg-white">
              <Layout className="mr-2 h-4 w-4" />
              Dry Run
            </Button>
            <Button onClick={handleRunWorkflow} className="bg-green-600 hover:bg-green-700 shadow-lg">
              <Play className="mr-2 h-4 w-4" />
              Run Workflow
            </Button>
          </div>
        )}

        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={readOnly ? undefined : (changes) => {
            onNodesChange(changes);
            // Only trigger onChange for position changes (drag), not selection
            if (changes.some(c => c.type === 'position' && 'dragging' in c && !c.dragging)) {
              onChange?.();
            }
          }}
          onEdgesChange={readOnly ? undefined : onEdgesChange}
          onConnect={onConnect}
          onPaneClick={onPaneClick}
          onNodeContextMenu={onNodeContextMenu}
          onEdgeContextMenu={onEdgeContextMenu}
          onNodeClick={onNodeClick}
          onNodeDoubleClick={onNodeDoubleClick}
          nodeTypes={nodeTypes}
          nodesDraggable={!readOnly}
          nodesConnectable={!readOnly}
          fitView
          defaultEdgeOptions={{
            type: 'default', // Bezier curve
            style: { strokeWidth: 2, stroke: '#94a3b8' },
            markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
          }}
        >
          <Controls showInteractive={false} />
          <MiniMap 
            nodeColor={(node) => {
              switch (node.data.status) {
                case 'success': return '#16a34a';
                case 'failed': return '#dc2626';
                case 'running': return '#3b82f6';
                default: return '#e2e8f0';
              }
            }}
            nodeStrokeWidth={3}
            zoomable
            pannable
          />
          <Background gap={12} size={1} />
          
          {/* Editor Toolbar */}
          <Panel position="bottom-left" className="bg-background/80 backdrop-blur-sm p-2 rounded-lg border shadow-sm flex gap-2">
            <Button variant="ghost" size="icon" onClick={handleAutoLayout} title="Auto Layout">
              <Layout className="h-4 w-4" />
            </Button>
            <Separator orientation="vertical" className="h-8" />
            <Button variant="ghost" size="icon" onClick={() => reactFlowInstance.fitView()} title="Fit View">
              <Maximize className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={() => reactFlowInstance.zoomIn()} title="Zoom In">
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={() => reactFlowInstance.zoomOut()} title="Zoom Out">
              <ZoomOut className="h-4 w-4" />
            </Button>
          </Panel>

          {menu && (
            <ContextMenu
              {...menu}
              onEdit={readOnly || menu.type === 'edge' ? undefined : handleEdit}
              onDelete={readOnly ? undefined : handleDelete}
              onViewResults={readOnly && menu.type === 'node' ? handleViewResults : undefined}
              onToggleOutput={readOnly || menu.type === 'edge' ? undefined : () => toggleOutput(menu.id)}
              isOutput={menu.type === 'node' ? nodes.find(n => n.id === menu.id)?.data.isOutput as boolean : undefined}
            />
          )}
        </ReactFlow>
      </div>

      {/* Right Sidebar - Node Details */}
      {selectedNode && (
        <NodeDetailPanel 
          node={selectedNode} 
          onClose={() => setSelectedNode(null)}
          onEdit={() => {
            if (readOnly) return;
            setEditingNodeId(selectedNode.id);
            setEditingNodeData(selectedNode.data);
            setIsConfigDialogOpen(true);
          }}
          executionStatus={executionStatus?.[selectedNode.id]}
        />
      )}
      
      {/* Dialogs */}
      <NodePaletteDialog 
        isOpen={isPaletteOpen} 
        onClose={() => setIsPaletteOpen(false)} 
        onNodeSelect={handleNodeAdd}
        nodes={availableNodes}
      />
      
      <NodeConfigDialog
        isOpen={isConfigDialogOpen}
        onClose={() => {
          setIsConfigDialogOpen(false);
          setPendingNodeToAdd(null);
          setEditingNodeId(null);
          setEditingNodeData(null);
        }}
        onConfirm={handleNodeConfigUpdate}
        nodeType={pendingNodeToAdd || (editingNodeId ? (nodes.find(n => n.id === editingNodeId)?.type || null) : null)}
        nodeDefinition={availableNodes.find(n => n.id === (pendingNodeToAdd || (editingNodeId ? nodes.find(node => node.id === editingNodeId)?.type : null)))}
        initialData={editingNodeData}
      />
    </div>
  );
};
