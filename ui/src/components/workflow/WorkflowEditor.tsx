import React, { useCallback, useRef, useState } from 'react';
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
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Play } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import { StartNode } from './nodes/StartNode';
import { EndNode } from './nodes/EndNode';
import { ProcessNode } from './nodes/ProcessNode';
import { NodeBrowser } from './NodeBrowser';
import { NodeConfigDialog } from './NodeConfigDialog';
import { ContextMenu } from './ContextMenu';
import { Button } from '../ui/button';
import { useAppStore } from '@/store/useAppStore';
import { getLayoutedElements } from '@/lib/workflow-utils';
import type { TaskGraphJson } from '@/types/task_graph_ir';

const nodeTypes = {
  start: StartNode,
  end: EndNode,
  process: ProcessNode,
  'load-molecule': StartNode,
  'optimize-geometry': ProcessNode,
  'calc-energy': ProcessNode,
  'run-md': ProcessNode,
  'save-results': EndNode,
};

const initialNodes: Node[] = [
  { id: '1', type: 'start', position: { x: 100, y: 100 }, data: { label: 'Start' } },
];

let id = 0;
const getId = () => `node_${id++}`;


interface WorkflowEditorProps {
  readOnly?: boolean;
  initialNodes?: Node[];
  initialEdges?: Edge[];
  initialGraph?: TaskGraphJson;
  executionStatus?: Record<string, 'success' | 'failed' | 'pending'>;
}

export const WorkflowEditor = ({ 
  readOnly = false, 
  initialNodes: propNodes, 
  initialEdges: propEdges,
  initialGraph,
  executionStatus 
}: WorkflowEditorProps) => {
  return (
    <ReactFlowProvider>
      <WorkflowEditorContent 
        readOnly={readOnly} 
        initialNodes={propNodes} 
        initialEdges={propEdges}
        initialGraph={initialGraph}
        executionStatus={executionStatus}
      />
    </ReactFlowProvider>
  );
};

const WorkflowEditorContent = ({ 
  readOnly, 
  initialNodes: propNodes, 
  initialEdges: propEdges,
  initialGraph,
  executionStatus 
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
  const [edges, setEdges, onEdgesChange] = useEdgesState(
    initialState.edges.map(edge => ({
      ...edge,
      style: executionStatus ? {
        stroke: executionStatus[edge.id] === 'success' ? '#16a34a' : 
                executionStatus[edge.id] === 'failed' ? '#dc2626' : '#94a3b8',
        strokeWidth: 2,
      } : undefined,
      animated: executionStatus ? executionStatus[edge.id] === 'pending' : false,
    }))
  );
  
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [selectedNodeType, setSelectedNodeType] = useState<string | null>(null);
  const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
  const [editingNodeData, setEditingNodeData] = useState<any>(null);

  const [menu, setMenu] = useState<{ id: string; top: number; left: number } | null>(null);
  const addExecution = useAppStore((state) => state.addExecution);
  const navigate = useNavigate();

  const handleRunWorkflow = () => {
    const newExecution = {
      id: Math.random().toString(36).substr(2, 9),
      name: `Workflow Run ${new Date().toLocaleTimeString()}`,
      status: 'Running' as const,
      date: new Date().toLocaleString(),
    };
    addExecution(newExecution);
    navigate('/executions');
  };

  const onConnect = useCallback(
    (params: Connection) => {
      if (readOnly) return;
      setEdges((eds) => addEdge(params, eds));
    },
    [setEdges, readOnly],
  );

  const handleNodeSelect = (type: string) => {
    if (readOnly) return;
    setSelectedNodeType(type);
    setEditingNodeId(null);
    setEditingNodeData(null);
    setIsDialogOpen(true);
  };

  const handleNodeAddOrUpdate = (data: any) => {
    if (readOnly) return;
    if (editingNodeId) {
      // Update existing node
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === editingNodeId) {
            return { ...node, data: { ...node.data, ...data } };
          }
          return node;
        })
      );
    } else if (selectedNodeType) {
      // Add new node
      const newNode: Node = {
        id: getId(),
        type: selectedNodeType,
        position: { x: Math.random() * 400 + 100, y: Math.random() * 400 + 100 },
        data: { ...data },
      };
      setNodes((nds) => nds.concat(newNode));
    }
    
    setIsDialogOpen(false);
    setSelectedNodeType(null);
    setEditingNodeId(null);
    setEditingNodeData(null);
  };

  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      event.preventDefault();

      if (!reactFlowWrapper.current) return;
      const pane = reactFlowWrapper.current.getBoundingClientRect();
      
      setMenu({
        id: node.id,
        top: event.clientY - pane.top,
        left: event.clientX - pane.left,
      });
    },
    []
  );

  const onPaneClick = useCallback(() => setMenu(null), []);

  const onNodeDoubleClick = useCallback((_event: React.MouseEvent, node: Node) => {
    if (readOnly) return;
    setEditingNodeId(node.id);
    setEditingNodeData(node.data);
    setSelectedNodeType(node.type || null);
    setIsDialogOpen(true);
  }, [readOnly]);

  const handleEdit = () => {
    if (!menu) return;
    const node = nodes.find((n) => n.id === menu.id);
    if (node) {
      setEditingNodeId(node.id);
      setEditingNodeData(node.data);
      setSelectedNodeType(node.type || null);
      setIsDialogOpen(true);
    }
    setMenu(null);
  };

  const handleDelete = () => {
    if (!menu) return;
    setNodes((nds) => nds.filter((node) => node.id !== menu.id));
    setEdges((eds) => eds.filter((edge) => edge.source !== menu.id && edge.target !== menu.id));
    setMenu(null);
  };

  const handleViewResults = () => {
    if (!menu) return;
    alert(`Viewing results for node ${menu.id}`);
    setMenu(null);
  };

  return (
    <div className="flex h-full w-full">
      {!readOnly && <NodeBrowser onNodeSelect={handleNodeSelect} />}
      <div className="flex-1 h-full relative" ref={reactFlowWrapper}>
        {!readOnly && (
          <div className="absolute top-4 right-4 z-10">
            <Button onClick={handleRunWorkflow} className="bg-green-600 hover:bg-green-700">
              <Play className="mr-2 h-4 w-4" />
              Run Workflow
            </Button>
          </div>
        )}
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={readOnly ? undefined : onNodesChange}
          onEdgesChange={readOnly ? undefined : onEdgesChange}
          onConnect={onConnect}
          onPaneClick={onPaneClick}
          onNodeContextMenu={onNodeContextMenu}
          onNodeDoubleClick={onNodeDoubleClick}
          nodeTypes={nodeTypes}
          nodesDraggable={!readOnly}
          nodesConnectable={!readOnly}
          fitView
        >
          <Controls />
          <MiniMap />
          <Background gap={12} size={1} />
          {menu && (
            <ContextMenu
              {...menu}
              onEdit={readOnly ? undefined : handleEdit}
              onDelete={readOnly ? undefined : handleDelete}
              onViewResults={readOnly ? handleViewResults : undefined}
            />
          )}
        </ReactFlow>
      </div>
      <NodeConfigDialog
        isOpen={isDialogOpen}
        onClose={() => setIsDialogOpen(false)}
        onConfirm={handleNodeAddOrUpdate}
        nodeType={selectedNodeType}
        initialData={editingNodeData}
      />
    </div>
  );
};
