// DataDisplay component for rendering JSON and metadata

import React from 'react';
import { formatJSON } from '@/utils/formatting';

export interface DataDisplayProps {
  title: string;
  data: unknown;
  maxHeight?: string;
}

export const DataDisplay: React.FC<DataDisplayProps> = ({
  title,
  data,
  maxHeight = 'max-h-64',
}) => {
  // Don't render if data is null, undefined, or empty object
  if (!data || (typeof data === 'object' && Object.keys(data).length === 0)) {
    return null;
  }

  return (
    <div>
      <h3 className="text-sm font-semibold mb-2">{title}</h3>
      <div className="bg-muted p-3 rounded-lg">
        <pre className={`text-xs overflow-auto ${maxHeight}`}>
          {formatJSON(data)}
        </pre>
      </div>
    </div>
  );
};
