import { Handle, Position } from '@xyflow/react';

export const EndNode = () => {
  return (
    <div className="w-32 h-16 bg-red-100 border-2 border-red-500 rounded-full flex items-center justify-center shadow-sm relative">
      <Handle type="target" position={Position.Top} className="w-3 h-3 bg-red-500" />
      <span className="font-semibold text-red-700">End</span>
    </div>
  );
};
