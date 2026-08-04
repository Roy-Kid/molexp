"""Land ad-hoc agent outputs onto a real molexp Run.

Chat often writes scripts / MolRec roots / figures via ``code_write`` /
``code_run`` without going through the workflow engine. Without landing,
the UI shows empty pending runs.

Landing is **workspace storage only**: open a RunContext, copy sources +
product files into the run as assets, optionally set small JSON headlines
via ``set_result``, and settle the lifecycle. Scientific series live in
**MolRec** records (observables / metrics JSONL / status) written by
molnex/molpy — the shared ``metrics/metrics.jsonl`` binding, not a
molexp-private format.
"""

from __future__ import annotations

import json
import mimetypes
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molexp.agent.ops.paths import safe_path

if TYPE_CHECKING:
    from molexp.workspace import Experiment, Project, Run, Workspace

__all__ = ["guess_mime", "land_run_outputs", "resolve_run"]


def guess_mime(path: Path) -> str | None:
    mime, _ = mimetypes.guess_type(path.name)
    return mime


def resolve_run(
    workspace_root: Path,
    *,
    project: str,
    experiment: str,
    run_id: str,
) -> Run:
    """Load a Run by project / experiment / run ids (or display names)."""
    from molexp.workspace import Workspace

    ws = Workspace(Path(workspace_root).resolve())
    proj = _get_project(ws, project)
    exp = _get_experiment(proj, experiment)
    return exp.get_run(run_id)


def _get_project(ws: Workspace, key: str) -> Project:
    try:
        return ws.get_project(key)
    except Exception:
        for p in ws.list_projects():
            if p.id == key or p.name == key:
                return p
        raise


def _get_experiment(proj: Project, key: str) -> Experiment:
    try:
        return proj.get_experiment(key)
    except Exception:
        for e in proj.list_experiments():
            if e.id == key or e.name == key:
                return e
        raise


def _infer_tags(rel: str, src: Path) -> dict[str, str]:
    tags: dict[str, str] = {"landed_from": rel}
    # Directory that looks like a MolRec root (has meta/) — tag for plugins.
    if src.is_dir() and (src / "meta").exists():
        tags["molrec"] = "true"
        sections: list[str] = []
        for name in (
            "system",
            "frame",
            "trajectory",
            "observables",
            "status",
            "metrics",
            "method",
        ):
            if (src / name).exists():
                sections.append(name)
        if sections:
            tags["molrec_sections"] = ",".join(sections)
    return tags


def land_run_outputs(
    workspace_root: Path,
    *,
    project: str,
    experiment: str,
    run_id: str,
    files: list[str] | None = None,
    sources: list[str] | None = None,
    results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach workspace-relative products to *run_id* and settle the run.

    Opens ``with run.start() as ctx`` so a pending/failed/cancelled run goes
    through the real lifecycle (running → succeeded). Live ``running`` and
    already-``succeeded`` runs refuse.

    Args:
        workspace_root: Session workspace root.
        project / experiment / run_id: Addressing triple (id or name).
        files: Workspace-relative paths to copy into ``run/artifacts/``
            (MolRec roots, figures, tables, JSON). Registered in the
            run asset manifest. Prefer MolRec directories for science.
        sources: Workspace-relative paths to copy into ``run/source/``
            (reviewable producer scripts).
        results: Optional small JSON headlines via ``ctx.set_result``
            (Overview Results) — not a substitute for MolRec observables.

    Returns:
        Summary dict: artifacts, sources, result keys, final status.
    """
    root = Path(workspace_root).resolve()
    run = resolve_run(root, project=project, experiment=experiment, run_id=run_id)

    status = str(run.status)
    if status == "running":
        raise ValueError(
            f"run {run.id} is live 'running' — cancel it first, or land into a new run"
        )
    if status == "succeeded":
        raise ValueError(
            f"run {run.id} already succeeded — create a new run and land into that "
            f"(or use resume/rerun for a fresh attempt)"
        )

    attached: list[str] = []
    sourced: list[str] = []
    result_keys: list[str] = []

    with run.start() as ctx:
        for rel in files or []:
            src = safe_path(root, rel)
            if not src.exists():
                raise FileNotFoundError(f"path not found: {rel}")
            name = src.name
            tags = _infer_tags(rel, src)
            if src.is_dir():
                # Copy directory tree into artifacts/<name>/
                dest = ctx.work_dir / "artifacts" / name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
                # Register a marker file so the manifest has an entry plugins can find.
                # Prefer registering the directory via a .molrec marker if present.
                marker = dest / "meta"
                if marker.exists():
                    # Point asset at the directory root by saving a small index.
                    index = dest / ".molexp-artifact.json"
                    index.write_text(
                        json.dumps({"kind": "molrec", "root": name, "tags": tags}),
                        encoding="utf-8",
                    )
                    ctx.artifact.save(
                        f"{name}/.molexp-artifact.json",
                        index,
                        mime="application/json",
                        tags=tags,
                    )
                else:
                    # Flat dir of loose files — register each file.
                    for child in dest.rglob("*"):
                        if child.is_file():
                            rel_child = child.relative_to(ctx.work_dir / "artifacts").as_posix()
                            ctx.artifact.save(
                                rel_child,
                                child,
                                mime=guess_mime(child),
                                tags=tags,
                            )
            else:
                # save() copies into artifacts/<name>; same-path is a no-op copy.
                ctx.artifact.save(name, src, mime=guess_mime(src), tags=tags)
            attached.append(name)
        for rel in sources or []:
            src = safe_path(root, rel)
            if not src.is_file():
                raise FileNotFoundError(f"source not found: {rel}")
            dest_dir = ctx.work_dir / "source"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if not (dest.exists() and src.resolve().samefile(dest.resolve())):
                import contextlib

                with contextlib.suppress(shutil.SameFileError):
                    shutil.copy2(src, dest)
            sourced.append(src.name)
            ctx.artifact.save(
                f"source/{src.name}",
                dest if dest.exists() else src,
                mime=guess_mime(src) or "text/x-python",
                tags={"role": "source", "landed_from": rel},
            )
        for key, value in (results or {}).items():
            ctx.set_result(str(key), value)
            result_keys.append(str(key))

    return {
        "run_id": run.id,
        "status": str(run.status),
        "artifacts": attached,
        "sources": sourced,
        "results": result_keys,
        "run_path": str(run.resolve()),
    }
