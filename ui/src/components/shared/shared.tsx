// Beautified shared UI components using shadcn-ui
// Enhanced with proper theming, spacing, and visual hierarchy

import React from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';

// ============================================================================
// StatCard - Individual statistic display with shadcn Card
// ============================================================================

export interface StatCardProps {
  value: string | number;
  label: string;
  icon?: React.ReactNode;
  className?: string;
}

export const StatCard: React.FC<StatCardProps> = ({ value, label, icon, className }) => (
  <Card className={cn('stat-card', className)}>
    <CardContent className="p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1">
          <div className="text-2xl font-bold tracking-tight">{value}</div>
          <div className="text-xs text-muted-foreground mt-1">{label}</div>
        </div>
        {icon && (
          <div className="text-muted-foreground opacity-60">
            {icon}
          </div>
        )}
      </div>
    </CardContent>
  </Card>
);

// ============================================================================
// StatGrid - Responsive grid layout for statistics
// ============================================================================

export interface StatGridProps {
  children: React.ReactNode;
  columns?: 2 | 3 | 4;
  className?: string;
}

export const StatGrid: React.FC<StatGridProps> = ({ children, columns = 2, className }) => {
  const gridClass = columns === 4 ? 'detail-grid-4' : 'detail-grid-2';
  return <div className={cn(gridClass, className)}>{children}</div>;
};

// ============================================================================
// DetailSection - Section wrapper with optional title and separator
// ============================================================================

export interface DetailSectionProps {
  title?: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  showSeparator?: boolean;
}

export const DetailSection: React.FC<DetailSectionProps> = ({ 
  title, 
  icon, 
  children, 
  className,
  showSeparator = false,
}) => (
  <div className={cn('space-y-3', className)}>
    {title && (
      <>
        <h3 className="text-sm font-semibold flex items-center gap-2">
          {icon}
          {title}
        </h3>
        {showSeparator && <Separator />}
      </>
    )}
    {children}
  </div>
);

// ============================================================================
// MetadataSection - Key-value metadata display with icons
// ============================================================================

export interface MetadataItem {
  icon: React.ReactNode;
  label: string;
  value: string | number | React.ReactNode;
}

export interface MetadataSectionProps {
  items: MetadataItem[];
  columns?: 1 | 2;
  className?: string;
}

export const MetadataSection: React.FC<MetadataSectionProps> = ({ 
  items, 
  columns = 2,
  className 
}) => {
  const gridClass = columns === 2 ? 'detail-grid-2' : 'space-y-4';
  
  return (
    <div className={cn(gridClass, className)}>
      {items.map((item, index) => (
        <div key={index} className="flex items-start gap-3">
          <div className="text-muted-foreground mt-0.5 flex-shrink-0">
            {item.icon}
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-detail-label mb-1">{item.label}</div>
            <div className="text-detail-value break-words">{item.value}</div>
          </div>
        </div>
      ))}
    </div>
  );
};

// ============================================================================
// EntityHeader - Consistent header for entities with Card
// ============================================================================

export interface EntityHeaderProps {
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}

export const EntityHeader: React.FC<EntityHeaderProps> = ({ 
  title, 
  subtitle, 
  badge, 
  actions,
  className 
}) => (
  <Card className={cn('bg-[var(--detail-panel-header-bg)]', className)}>
    <CardHeader>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0 space-y-2">
          <CardTitle className="text-detail-title break-words">{title}</CardTitle>
          {badge && <div>{badge}</div>}
          {subtitle && (
            <CardDescription className="text-detail-subtitle">
              {subtitle}
            </CardDescription>
          )}
        </div>
        {actions && <div className="flex-shrink-0">{actions}</div>}
      </div>
    </CardHeader>
  </Card>
);

// ============================================================================
// CodeBlock - Display code or hashes with ScrollArea
// ============================================================================

export interface CodeBlockProps {
  code: string;
  label?: string;
  className?: string;
  maxHeight?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ 
  code, 
  label, 
  className,
  maxHeight = 'max-h-32'
}) => (
  <div className={cn('space-y-2', className)}>
    {label && (
      <div className="text-detail-label">{label}</div>
    )}
    <Card className="code-block">
      <CardContent className="p-0">
        <ScrollArea className={maxHeight}>
          <code className="text-detail-mono block p-3 break-all">{code}</code>
        </ScrollArea>
      </CardContent>
    </Card>
  </div>
);

// ============================================================================
// TagList - Display tags with consistent styling
// ============================================================================

export interface TagListProps {
  tags: string[];
  icon?: React.ReactNode;
  title?: string;
  className?: string;
}

export const TagList: React.FC<TagListProps> = ({ tags, icon, title, className }) => {
  if (!tags || tags.length === 0) return null;
  
  return (
    <DetailSection title={title} icon={icon} className={className}>
      <div className="flex flex-wrap gap-2">
        {tags.map((tag) => (
          <span 
            key={tag} 
            className="px-2.5 py-1 bg-accent text-accent-foreground rounded-md text-xs font-medium"
          >
            {tag}
          </span>
        ))}
      </div>
    </DetailSection>
  );
};

// ============================================================================
// ListCard - Card wrapper for lists with consistent styling
// ============================================================================

export interface ListCardProps {
  children: React.ReactNode;
  className?: string;
}

export const ListCard: React.FC<ListCardProps> = ({ children, className }) => (
  <Card className={cn('hover:shadow-sm transition-shadow', className)}>
    <CardContent className="p-3">
      {children}
    </CardContent>
  </Card>
);

// ============================================================================
// EmptyState - Display when no data
// ============================================================================

export interface EmptyStateProps {
  icon?: React.ReactNode;
  message: string;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ icon, message, className }) => (
  <Card className={cn('border-dashed', className)}>
    <CardContent className="flex flex-col items-center justify-center py-12 text-center">
      {icon && <div className="mb-3 text-muted-foreground opacity-50">{icon}</div>}
      <p className="text-sm text-muted-foreground">{message}</p>
    </CardContent>
  </Card>
);

// ============================================================================
// LoadingState - Display during loading with skeletons
// ============================================================================

export interface LoadingStateProps {
  message?: string;
  className?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ message, className }) => (
  <div className={cn('space-y-4', className)}>
    <Card>
      <CardHeader>
        <Skeleton className="h-8 w-3/4" />
        <Skeleton className="h-4 w-1/2 mt-2" />
      </CardHeader>
    </Card>
    <div className="detail-grid-2">
      <Skeleton className="h-24" />
      <Skeleton className="h-24" />
    </div>
    <Skeleton className="h-32" />
    {message && (
      <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>{message}</span>
      </div>
    )}
  </div>
);

// ============================================================================
// ErrorState - Display errors with retry option
// ============================================================================

export interface ErrorStateProps {
  error: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({ error, onRetry, className }) => (
  <Card className={cn('border-destructive/50 bg-destructive/5', className)}>
    <CardContent className="p-6">
      <div className="flex items-start gap-3">
        <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
        <div className="flex-1 space-y-2">
          <p className="font-semibold text-sm text-destructive">Error loading details</p>
          <p className="text-xs text-muted-foreground">{error}</p>
          {onRetry && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onRetry}
              className="mt-3 bg-destructive/10 text-destructive hover:bg-destructive/20 hover:text-destructive"
            >
              Retry
            </Button>
          )}
        </div>
      </div>
    </CardContent>
  </Card>
);

// ============================================================================
// StaleIndicator - Visual indicator for stale data
// ============================================================================

export interface StaleIndicatorProps {
  className?: string;
}

export const StaleIndicator: React.FC<StaleIndicatorProps> = ({ className }) => (
  <div className={cn(
    'absolute top-0 left-0 right-0 h-1 bg-primary/30 animate-pulse',
    className
  )} />
);
