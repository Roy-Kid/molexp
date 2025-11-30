"""Workspace validation and integrity checking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .workspace import Workspace


@dataclass
class Issue:
    """Represents a validation issue."""

    severity: str  # "error", "warning", "info"
    category: str  # "structure", "metadata", "assets", "orphaned"
    message: str
    path: Path | None = None
    fixable: bool = False


@dataclass
class ValidationReport:
    """Validation report with issues and statistics."""

    issues: list[Issue]
    total_projects: int = 0
    total_experiments: int = 0
    total_runs: int = 0
    total_assets: int = 0
    orphaned_assets: int = 0

    @property
    def has_errors(self) -> bool:
        """Check if report has any errors."""
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if report has any warnings."""
        return any(issue.severity == "warning" for issue in self.issues)

    def summary(self) -> str:
        """Get summary string."""
        errors = sum(1 for i in self.issues if i.severity == "error")
        warnings = sum(1 for i in self.issues if i.severity == "warning")
        
        if not self.issues:
            return "✓ Workspace is valid"
        
        parts = []
        if errors:
            parts.append(f"{errors} error(s)")
        if warnings:
            parts.append(f"{warnings} warning(s)")
        
        return f"✗ Found {', '.join(parts)}"


class WorkspaceValidator:
    """Validate workspace integrity and consistency."""

    def __init__(self, workspace: Workspace) -> None:
        """Initialize workspace validator.
        
        Args:
            workspace: Workspace to validate
        """
        self.workspace = workspace

    def validate(self, verbose: bool = False) -> ValidationReport:
        """Run all validation checks.
        
        Args:
            verbose: Include informational messages
            
        Returns:
            Validation report
        """
        issues: list[Issue] = []

        # Check structure
        issues.extend(self.check_structure())

        # Check metadata
        issues.extend(self.check_metadata())

        # Check assets
        issues.extend(self.check_assets())

        # Find orphaned assets
        orphaned = self.find_orphaned_assets()
        if orphaned:
            issues.append(
                Issue(
                    severity="warning",
                    category="orphaned",
                    message=f"Found {len(orphaned)} orphaned asset(s) not referenced by any run",
                    fixable=True,
                )
            )

        # Gather statistics
        projects = self.workspace.list_projects()
        total_experiments = 0
        total_runs = 0
        
        for project in projects:
            experiments = self.workspace.list_experiments(project.project_id)
            total_experiments += len(experiments)
            
            for experiment in experiments:
                runs = self.workspace.list_runs(project.project_id, experiment.experiment_id)
                total_runs += len(runs)

        assets = self.workspace.list_assets()

        report = ValidationReport(
            issues=issues,
            total_projects=len(projects),
            total_experiments=total_experiments,
            total_runs=total_runs,
            total_assets=len(assets),
            orphaned_assets=len(orphaned),
        )

        return report

    def check_structure(self) -> list[Issue]:
        """Check directory structure.
        
        Returns:
            List of issues found
        """
        issues: list[Issue] = []

        # Check required directories
        required_dirs = [
            self.workspace.root / "projects",
            self.workspace.root / "assets",
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                issues.append(
                    Issue(
                        severity="error",
                        category="structure",
                        message=f"Required directory missing: {dir_path.name}",
                        path=dir_path,
                        fixable=True,
                    )
                )
            elif not dir_path.is_dir():
                issues.append(
                    Issue(
                        severity="error",
                        category="structure",
                        message=f"Path exists but is not a directory: {dir_path.name}",
                        path=dir_path,
                        fixable=False,
                    )
                )

        return issues

    def check_metadata(self) -> list[Issue]:
        """Check metadata file consistency.
        
        Returns:
            List of issues found
        """
        issues: list[Issue] = []

        projects_dir = self.workspace.root / "projects"
        if not projects_dir.exists():
            return issues

        # Check each project
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            # Check project.json
            project_json = project_dir / "project.json"
            if not project_json.exists():
                issues.append(
                    Issue(
                        severity="error",
                        category="metadata",
                        message=f"Missing project.json in {project_dir.name}",
                        path=project_json,
                        fixable=False,
                    )
                )
                continue

            # Validate project JSON
            try:
                with open(project_json) as f:
                    project_data = json.load(f)
                
                # Check required fields
                required_fields = ["project_id", "name", "created_at"]
                for field in required_fields:
                    if field not in project_data:
                        issues.append(
                            Issue(
                                severity="error",
                                category="metadata",
                                message=f"Missing required field '{field}' in {project_dir.name}/project.json",
                                path=project_json,
                                fixable=False,
                            )
                        )
            except json.JSONDecodeError as e:
                issues.append(
                    Issue(
                        severity="error",
                        category="metadata",
                        message=f"Invalid JSON in {project_dir.name}/project.json: {e}",
                        path=project_json,
                        fixable=False,
                    )
                )
                continue

            # Check experiments
            experiments_dir = project_dir / "experiments"
            if experiments_dir.exists():
                for exp_dir in experiments_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue

                    # Check experiment.json
                    exp_json = exp_dir / "experiment.json"
                    if not exp_json.exists():
                        issues.append(
                            Issue(
                                severity="error",
                                category="metadata",
                                message=f"Missing experiment.json in {project_dir.name}/{exp_dir.name}",
                                path=exp_json,
                                fixable=False,
                            )
                        )
                        continue

                    # Check runs
                    runs_dir = exp_dir / "runs"
                    if runs_dir.exists():
                        for run_dir in runs_dir.iterdir():
                            if not run_dir.is_dir():
                                continue

                            # Check run.json
                            run_json = run_dir / "run.json"
                            if not run_json.exists():
                                issues.append(
                                    Issue(
                                        severity="error",
                                        category="metadata",
                                        message=f"Missing run.json in {project_dir.name}/{exp_dir.name}/{run_dir.name}",
                                        path=run_json,
                                        fixable=False,
                                    )
                                )

        return issues

    def check_assets(self) -> list[Issue]:
        """Check asset repository consistency.
        
        Returns:
            List of issues found
        """
        issues: list[Issue] = []

        assets_dir = self.workspace.root / "assets"
        if not assets_dir.exists():
            return issues

        # Check asset metadata vs actual files
        assets = self.workspace.list_assets()
        
        for asset in assets:
            # Check if asset files exist
            if hasattr(asset, "files"):
                for file_info in asset.files:
                    file_path = assets_dir / file_info.path
                    if not file_path.exists():
                        issues.append(
                            Issue(
                                severity="error",
                                category="assets",
                                message=f"Asset file missing: {file_info.path} (asset: {asset.asset_id[:8]}...)",
                                path=file_path,
                                fixable=False,
                            )
                        )

        return issues

    def find_orphaned_assets(self) -> list[str]:
        """Find assets not referenced by any run.
        
        Returns:
            List of orphaned asset IDs
        """
        # Get all assets
        all_assets = {asset.asset_id for asset in self.workspace.list_assets()}

        # Get all referenced assets
        referenced_assets = set()
        
        projects = self.workspace.list_projects()
        for project in projects:
            experiments = self.workspace.list_experiments(project.project_id)
            for experiment in experiments:
                runs = self.workspace.list_runs(project.project_id, experiment.experiment_id)
                for run in runs:
                    # Get asset refs
                    refs = self.workspace.get_asset_refs(
                        project.project_id, experiment.experiment_id, run.run_id
                    )
                    if refs:
                        referenced_assets.update(ref.asset_id for ref in refs.inputs)
                        referenced_assets.update(ref.asset_id for ref in refs.outputs)

        # Find orphaned
        orphaned = all_assets - referenced_assets
        return list(orphaned)

    def fix_issues(self, issues: list[Issue]) -> list[Issue]:
        """Attempt to fix issues automatically.
        
        Args:
            issues: List of issues to fix
            
        Returns:
            List of issues that could not be fixed
        """
        unfixed: list[Issue] = []

        for issue in issues:
            if not issue.fixable:
                unfixed.append(issue)
                continue

            try:
                if issue.category == "structure" and issue.path:
                    # Create missing directory
                    issue.path.mkdir(parents=True, exist_ok=True)
                elif issue.category == "orphaned":
                    # Could implement orphaned asset cleanup here
                    # For now, just mark as unfixed
                    unfixed.append(issue)
                else:
                    unfixed.append(issue)
            except Exception:
                unfixed.append(issue)

        return unfixed
