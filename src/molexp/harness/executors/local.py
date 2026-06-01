"""``LocalExecutor`` — real subprocess execution with stdout/stderr capture.

Honors ``CommandSpec.cwd`` / ``env`` / ``timeout_s`` / ``expected_outputs``.
Captures stdout/stderr text and persists them as artifacts of kinds
``stdout`` / ``stderr``. On timeout, persists whatever was captured
before the kill and returns ``exit_code=-1`` with
``metadata["timeout"]="true"``.

Non-blocking: uses :func:`asyncio.create_subprocess_exec` so the event
loop stays responsive while a child process runs (the previous
``subprocess.run`` implementation blocked every concurrent task in the
loop for ``timeout_s`` seconds).

Strict env semantics: ``spec.env`` is forwarded verbatim — an empty
dict means *empty environment*, not "inherit parent env". ``None``
means inherit. The previous implementation conflated empty-dict with
None and so the user's `env={}` silently inherited the parent env.

Expected-output collection: after the process exits, every entry in
``spec.expected_outputs`` is resolved relative to ``spec.cwd`` and, if
present and within ``spec.cwd`` (no path traversal), persisted as an
``output_file`` artifact and added to ``CommandResult.output_artifacts``.
Missing or escape-attempting paths are recorded in ``result.metadata``
under ``missing_outputs`` / ``escaped_outputs`` (comma-joined) so audit
consumers can flag them; the executor does not fail the command for
missing outputs (that's the validator's job downstream).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from molexp.harness.schemas import CommandResult, CommandSpec
from molexp.harness.store.artifact_store import ArtifactStore

__all__ = ["LocalExecutor"]


class LocalExecutor:
    """Asynchronous subprocess executor."""

    async def execute(
        self,
        spec: CommandSpec,
        *,
        artifact_store: ArtifactStore,
    ) -> CommandResult:
        started = datetime.now(tz=UTC)
        stdout_bytes = b""
        stderr_bytes = b""
        timed_out = False
        exit_code: int

        proc = await asyncio.create_subprocess_exec(
            *spec.cmd,
            cwd=spec.cwd,
            env=None if spec.env is None else dict(spec.env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=spec.timeout_s
            )
            exit_code = proc.returncode if proc.returncode is not None else -1
        except TimeoutError:
            timed_out = True
            exit_code = -1
            # Kill the child and drain whatever was buffered. We swallow
            # any secondary error here — the timeout itself is the
            # outcome we want to report.
            with _suppress_process_lookup():
                proc.kill()
            try:
                stdout_bytes, stderr_bytes = await proc.communicate()
            except Exception:
                stdout_bytes, stderr_bytes = b"", b""

        ended = datetime.now(tz=UTC)

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        stdout_ref = artifact_store.put_text(
            kind="stdout",
            text=stdout_text,
            created_by="LocalExecutor",
            parent_ids=[],
        )
        stderr_ref = artifact_store.put_text(
            kind="stderr",
            text=stderr_text,
            created_by="LocalExecutor",
            parent_ids=[],
        )

        # Resolve expected outputs after the process has exited.
        output_refs: list = []
        missing: list[str] = []
        escaped: list[str] = []
        if spec.expected_outputs:
            cwd_path = Path(spec.cwd).resolve()
            for rel in spec.expected_outputs:
                candidate = (cwd_path / rel).resolve()
                # Defense in depth: refuse to ingest anything outside cwd
                # so a malicious or buggy command can't trick the harness
                # into hashing /etc/passwd or similar.
                if not _is_within(candidate, cwd_path):
                    escaped.append(rel)
                    continue
                if not candidate.exists() or not candidate.is_file():
                    missing.append(rel)
                    continue
                ref = artifact_store.put_file(
                    kind="output_file",
                    path=candidate,
                    created_by="LocalExecutor",
                    parent_ids=[],
                )
                output_refs.append(ref)

        metadata = {"executor": "LocalExecutor"}
        if timed_out:
            metadata["timeout"] = "true"
        if missing:
            metadata["missing_outputs"] = ",".join(missing)
        if escaped:
            metadata["escaped_outputs"] = ",".join(escaped)

        return CommandResult(
            exit_code=exit_code,
            started_at=started,
            ended_at=ended,
            stdout_artifact=stdout_ref,
            stderr_artifact=stderr_ref,
            output_artifacts=output_refs,
            metadata=metadata,
        )


def _is_within(candidate: Path, parent: Path) -> bool:
    """True iff ``candidate`` is the same as or a descendant of ``parent``.

    Both inputs MUST already be resolved (i.e. symlink + ``..`` collapsed),
    otherwise a relative ``..`` segment can sneak past the prefix check.
    """
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


class _suppress_process_lookup:
    """Context manager that swallows ``ProcessLookupError`` on kill races.

    The child may have already exited between the timeout firing and the
    ``proc.kill()`` call; that race is normal and not worth reporting.
    """

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool:
        return exc_type is not None and issubclass(exc_type, ProcessLookupError)
