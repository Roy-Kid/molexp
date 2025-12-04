"""Test script for indexed folder system."""

from pathlib import Path
from datetime import datetime

from molexp.workspace import Workspace
from molexp.workspace.scanner import FolderScanner
from molexp.repositories.indexed import IndexFileManager


def test_indexed_folder_system():
    """Test the indexed folder system functionality."""
    
    # Get workspace
    workspace_path = Path.cwd()
    workspace = Workspace.from_path(workspace_path)
    
    print(f"Testing indexed folder system in: {workspace.root}\n")
    
    # Test 1: List existing projects
    print("=" * 60)
    print("Test 1: List existing projects")
    print("=" * 60)
    projects = workspace.list_projects()
    print(f"Found {len(projects)} projects:")
    for project in projects:
        print(f"  - {project.name} ({project.project_id})")
        print(f"    Kind: {project.kind}")
        print(f"    Schema version: {project.schema_version}")
        print(f"    Created: {project.created_at}")
        print(f"    Updated: {project.updated_at}")
    print()
    
    # Test 2: Scan workspace for indexed folders
    print("=" * 60)
    print("Test 2: Scan workspace for indexed folders")
    print("=" * 60)
    scanner = FolderScanner(workspace.root)
    entities = scanner.scan_workspace()
    print(f"Found {len(entities)} indexed entities:")
    for entity_info in entities:
        print(f"  - {entity_info['kind']}: {entity_info['path']}")
        print(f"    Schema version: {entity_info['entity'].schema_version}")
    print()
    
    # Test 3: Detect entity kind
    print("=" * 60)
    print("Test 3: Detect entity kind for specific folders")
    print("=" * 60)
    
    # Test project folder
    if projects:
        project_path = workspace.root / "projects" / projects[0].project_id
        kind = IndexFileManager.detect_entity_kind(project_path)
        print(f"  Folder: {project_path.name}")
        print(f"  Detected kind: {kind}")
        
        # Test if it's indexed
        is_indexed = scanner.is_indexed_folder(project_path)
        print(f"  Is indexed: {is_indexed}")
    print()
    
    # Test 4: Read index file
    print("=" * 60)
    print("Test 4: Read and parse index files")
    print("=" * 60)
    if projects:
        project_path = workspace.root / "projects" / projects[0].project_id
        from molexp.models import Project
        
        project = IndexFileManager.read_index(project_path, "project", Project)
        if project:
            print(f"  Successfully read project: {project.name}")
            print(f"  Project ID: {project.id}")
            print(f"  Kind: {project.kind}")
            print(f"  Schema version: {project.schema_version}")
    print()
    
    # Test 5: Check experiments
    print("=" * 60)
    print("Test 5: Check experiments in projects")
    print("=" * 60)
    if projects:
        experiments = workspace.list_experiments(projects[0].project_id)
        print(f"Found {len(experiments)} experiments in '{projects[0].name}':")
        for exp in experiments:
            print(f"  - {exp.name} ({exp.experiment_id})")
            print(f"    Kind: {exp.kind}")
            print(f"    Schema version: {exp.schema_version}")
    print()
    
    print("=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_indexed_folder_system()
