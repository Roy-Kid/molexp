// Reusable InfoItem component for displaying labeled information

import React from 'react';

export interface InfoItemProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
}

export const InfoItem: React.FC<InfoItemProps> = ({ icon, label, value }) => (
  <div className="flex items-start gap-2">
    <div className="text-muted-foreground mt-0.5">{icon}</div>
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-sm font-medium">{value}</div>
    </div>
  </div>
);
