/**
 * File utility functions for file preview and editing.
 * Consolidates duplicated logic from FilePreview and FileEditor components.
 */

/**
 * Get the file extension from a filename.
 * Returns lowercase extension with leading dot (e.g., ".md").
 * 
 * @param filename - The filename to extract extension from
 * @returns The lowercase extension with dot, or empty string if none
 */
export function getFileExtension(filename: string): string {
    const lastDot = filename.lastIndexOf('.');
    if (lastDot === -1 || lastDot === 0) {
        return '';
    }
    return filename.slice(lastDot).toLowerCase();
}

/**
 * Get the Monaco editor language ID for a given filename.
 * Maps file extensions to Monaco language identifiers for syntax highlighting.
 * 
 * @param filename - The filename to determine language for
 * @returns The Monaco language ID
 */
export function getMonacoLanguage(filename: string): string {
    const ext = getFileExtension(filename);

    switch (ext) {
        case '.py':
        case '.ipynb':
            return 'python';
        case '.ts':
        case '.tsx':
            return 'typescript';
        case '.js':
        case '.jsx':
            return 'javascript';
        case '.json':
            return 'json';
        case '.yml':
        case '.yaml':
            return 'yaml';
        case '.md':
        case '.markdown':
        case '.mdx':
            return 'markdown';
        case '.css':
            return 'css';
        case '.html':
        case '.htm':
            return 'html';
        case '.xml':
            return 'xml';
        case '.sh':
        case '.bash':
            return 'shell';
        case '.sql':
            return 'sql';
        case '.c':
        case '.h':
            return 'c';
        case '.cpp':
        case '.hpp':
        case '.cc':
            return 'cpp';
        case '.java':
            return 'java';
        case '.go':
            return 'go';
        case '.rs':
            return 'rust';
        case '.toml':
            return 'toml';
        case '.ini':
        case '.cfg':
            return 'ini';
        default:
            return 'plaintext';
    }
}
