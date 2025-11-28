import { Handle, Position } from '@xyflow/react';

export const StartNode = () => {
  return (
    <div className="w-32 h-16 bg-green-100 border-2 border-green-500 rounded-full flex items-center justify-center shadow-sm relative">
      <span className="font-semibold text-green-700">Start</span>
      <Handle type="source" position={Position.Bottom} className="w-3 h-3 bg-green-500" />
    </div>
  );
};
