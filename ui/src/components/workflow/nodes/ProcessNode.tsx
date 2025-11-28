import { Handle, Position } from '@xyflow/react';

export const ProcessNode = ({ data }: { data: { label: string } }) => {
  return (
    <div className="w-40 h-20 bg-blue-50 border-2 border-blue-400 rounded-lg flex items-center justify-center shadow-sm relative">
      <Handle type="target" position={Position.Top} className="w-3 h-3 bg-blue-400" />
      <span className="font-medium text-blue-900">{data.label}</span>
      <Handle type="source" position={Position.Bottom} className="w-3 h-3 bg-blue-400" />
    </div>
  );
};
