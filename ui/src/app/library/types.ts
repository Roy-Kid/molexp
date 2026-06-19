// Wire shapes for the /api/library endpoints. Hand-written (not generated)
// because libraryApi uses raw fetch — mirrors getServedWorkspaces et al.

export interface LibraryNoteEntry {
  asset_id: string;
  title: string;
  path: string;
  summary: string;
  tags: string[];
  refs: string[];
  updated_at: string | null;
}

export interface LibraryReference {
  key: string;
  title: string;
  authors: string[];
  year: number | null;
  venue: string | null;
  arxiv: string | null;
  doi: string | null;
  url: string | null;
  tags: string[];
  note: string;
  pdf_asset_id: string | null;
  pdf_path: string | null;
  source: string | null;
  source_key: string | null;
  added_at: string;
}

export interface LibrarySource {
  kind: string;
  path: string;
  count: number;
  last_synced: string;
}

export interface LibraryIndex {
  schema_version: number;
  scope: string;
  generated_at: string;
  notes: LibraryNoteEntry[];
  references: LibraryReference[];
}

// Which scope's library to address. Empty -> workspace scope.
export interface LibraryScope {
  projectId?: string;
  experimentId?: string;
  runId?: string;
}

export interface AddNotePayload {
  title: string;
  content: string;
  summary?: string;
  tags?: string[];
  refs?: string[];
}
