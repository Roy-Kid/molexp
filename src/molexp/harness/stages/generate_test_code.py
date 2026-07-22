"""``GenerateTestCode`` — ask the gateway to emit pytest source for the workflow.

Fans the codegen out to **one gateway call per task** (``test_code_file_writer``)
and assembles the per-file results into a single ``test_source`` artifact. The
whole-suite ``test_code_writer`` call on the HEAVY (v4-pro) tier times out — a
multi-file pytest program is a large structured output and a single pro call can
exceed the provider timeout (~12 min observed). One small per-task call each
stays well under it, so every write can run on pro. A workflow with a single
task (or an unparseable spec bundle) falls back to the original single
``test_code_writer`` call.

On a :class:`RepairLoop` retry the previous attempt's ``test_code_feedback``
(the byte-compile / structural violations from :class:`ValidateTestSource`) is
threaded into every per-file call, mirroring :class:`GenerateWorkflowSource`.
"""

from __future__ import annotations

import asyncio
import re
from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.schemas import (
    AgentCallSpec,
    PlanArtifactRef,
    TestSource,
    TestSpecBundle,
)
from molexp.harness.schemas.workflow_source import GeneratedFile
from molexp.harness.stages._resolve import feedback_inputs, require_latest

__all__ = ["GenerateTestCode"]


class GenerateTestCode(Stage):
    """Generate pytest source realizing the TestSpec via gateway (per-file)."""

    name: ClassVar[str] = "generate_test_code"

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        if ctx.agent_gateway is None:
            raise StageExecutionError("GenerateTestCode requires ctx.agent_gateway to be set")
        test_spec = require_latest(ctx, "test_spec", stage=self.name)
        workflow_source = require_latest(ctx, "workflow_source", stage=self.name)
        feedback = feedback_inputs(ctx, "test_code_feedback")

        specs = self._parse_specs(ctx, test_spec.id)
        # Single task (or unparseable bundle) → the whole-suite call is small
        # enough; keep the original one-shot path.
        if len(specs) <= 1:
            spec = AgentCallSpec(
                agent_name="test_code_writer",
                input_artifact_ids=[test_spec.id, workflow_source.id, *feedback],
                output_schema=TestSource.model_json_schema(),
            )
            return (await ctx.agent_gateway.call(spec)).output_artifact

        return await self._generate_per_file(
            ctx,
            bundle_ref=test_spec,
            specs=specs,
            workflow_source_id=workflow_source.id,
            feedback=feedback,
        )

    def _parse_specs(self, ctx: HarnessRunContext, test_spec_id: str) -> list:
        """Parse the ``test_spec`` bundle into its per-task specs (or [])."""
        try:
            bundle = TestSpecBundle.from_artifact(ctx.artifact_store.get(test_spec_id))
        except Exception:
            return []
        return list(bundle.specs)

    async def _generate_per_file(
        self,
        ctx: HarnessRunContext,
        *,
        bundle_ref: PlanArtifactRef,
        specs: list,
        workflow_source_id: str,
        feedback: list[str],
    ) -> PlanArtifactRef:
        """One ``test_code_file_writer`` call per task (concurrent), then merged."""
        bundle = TestSpecBundle.from_artifact(ctx.artifact_store.get(bundle_ref.id))

        async def one(spec) -> tuple[str, list[GeneratedFile]]:
            """Return (partial_artifact_id, files) for one task's test module."""
            slice_bundle = TestSpecBundle(
                id=f"{bundle.id}-{spec.target_task_id or spec.id}",
                bound_workflow_id=bundle.bound_workflow_id,
                specs=[spec],
            )
            slice_ref = ctx.artifact_store.put_json(
                kind="test_spec_slice",
                obj=slice_bundle.model_dump(mode="json"),
                created_by=f"stage:{self.name}",
                parent_ids=[bundle_ref.id],
            )
            call = AgentCallSpec(
                agent_name="test_code_file_writer",
                input_artifact_ids=[slice_ref.id, workflow_source_id, *feedback],
                output_schema=TestSource.model_json_schema(),
            )
            result = await ctx.agent_gateway.call(call)
            partial = TestSource.model_validate_json(
                ctx.artifact_store.get(result.output_artifact.id)
            )
            return result.output_artifact.id, self._files_of(partial, spec)

        results = await asyncio.gather(*(one(s) for s in specs))
        merged: dict[str, GeneratedFile] = {}
        partial_ids: list[str] = []
        for partial_id, gfs in results:
            partial_ids.append(partial_id)
            for gf in gfs:
                merged[gf.path] = gf  # last write wins on a path collision

        if not merged:
            raise StageExecutionError(
                f"stage {self.name!r}: per-file test generation produced no files"
            )

        files = list(merged.values())
        # `source` MUST be empty in multi-file mode: a non-empty entry source
        # makes MaterializeExecution write an extra `{module_name}/__init__.py`
        # package, which collides with the same-stem `tests/test_<task>.py`
        # ("import file mismatch" at pytest collection). Everything runs from
        # `files`; `module_name` is only a label here (unused when files exist).
        assembled = TestSource(
            source="",
            module_name="test_suite",
            test_spec_id=bundle.id,
            bound_workflow_id=bundle.bound_workflow_id,
            files=files,
        )
        return ctx.artifact_store.put_json(
            kind="test_source",
            obj=assembled.model_dump(mode="json"),
            created_by=f"stage:{self.name}",
            parent_ids=[bundle_ref.id, workflow_source_id, *partial_ids, *feedback],
        )

    @staticmethod
    def _files_of(partial: TestSource, spec) -> list[GeneratedFile]:
        """The per-file result's test modules (synthesize one if single-file)."""
        if partial.files:
            return list(partial.files)
        token = re.sub(r"\W", "_", spec.target_task_id or spec.id)
        return [GeneratedFile(path=f"tests/test_{token}.py", source=partial.source)]
