// StatusBadge component with proper typing and configuration

import React from 'react';
import { CheckCircle, XCircle, Loader, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { RunStatus } from '@/types/domain';

export interface StatusBadgeProps {
  status: RunStatus | string;
}

interface StatusConfig {
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  bg: string;
}

const STATUS_CONFIG: Record<string, StatusConfig> = {
  succeeded: {
    icon: CheckCircle,
    color: 'text-green-500',
    bg: 'bg-green-500/10',
  },
  failed: {
    icon: XCircle,
    color: 'text-red-500',
    bg: 'bg-red-500/10',
  },
  running: {
    icon: Loader,
    color: 'text-blue-500',
    bg: 'bg-blue-500/10',
  },
  pending: {
    icon: Clock,
    color: 'text-gray-500',
    bg: 'bg-gray-500/10',
  },
};

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  if (!status) {
    return null;
  }

  const config = STATUS_CONFIG[status.toLowerCase()] || STATUS_CONFIG.pending;
  const Icon = config.icon;

  return (
    <div
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium',
        config.bg,
        config.color
      )}
    >
      <Icon
        className={cn(
          'h-3.5 w-3.5',
          status.toLowerCase() === 'running' && 'animate-spin'
        )}
      />
      {status}
    </div>
  );
};
