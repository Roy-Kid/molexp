// Formatting utilities for DetailPanel components

/**
 * Format a date string or Date object to localized date string
 * Safely handles null, undefined, and invalid dates
 */
export function formatDate(date: string | Date | undefined | null): string {
    if (!date) return 'N/A';
    try {
        const dateObj = typeof date === 'string' ? new Date(date) : date;
        if (isNaN(dateObj.getTime())) return 'Invalid Date';
        return dateObj.toLocaleDateString();
    } catch {
        return 'Invalid Date';
    }
}

/**
 * Format a date string or Date object to localized date-time string
 * Safely handles null, undefined, and invalid dates
 */
export function formatDateTime(date: string | Date | undefined | null): string {
    if (!date) return 'N/A';
    try {
        const dateObj = typeof date === 'string' ? new Date(date) : date;
        if (isNaN(dateObj.getTime())) return 'Invalid Date';
        return dateObj.toLocaleString();
    } catch {
        return 'Invalid Date';
    }
}

/**
 * Format bytes to human-readable size
 * Handles null, undefined, and zero values safely
 */
export function formatBytes(bytes: number | undefined | null): string {
    if (bytes == null || bytes === 0) return '0 B';
    if (bytes < 0) return 'Invalid Size';

    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    // Clamp to valid array index
    const sizeIndex = Math.min(i, sizes.length - 1);

    return `${Math.round((bytes / Math.pow(k, sizeIndex)) * 100) / 100} ${sizes[sizeIndex]}`;
}

/**
 * Truncate git commit hash to short form (7 characters)
 * Safely handles null and undefined
 */
export function formatGitCommit(commit: string | undefined | null): string {
    if (!commit) return 'N/A';
    return commit.slice(0, 7);
}

/**
 * Format data as pretty-printed JSON
 * Safely handles any input and catches serialization errors
 */
export function formatJSON(data: unknown): string {
    try {
        return JSON.stringify(data, null, 2);
    } catch {
        return 'Invalid JSON';
    }
}
