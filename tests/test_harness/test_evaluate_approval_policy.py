"""Tests for evaluate_approval_policy + make_final_report_approval_request (Phase 6).

One focused test per intent + clean baseline + ordering invariant +
final-report helper.
"""

from __future__ import annotations


def _clean_bound_workflow(*, backend: str = "local"):
    """Local backend, no agent_inferred params, no overwrite flag, 1h/8GB."""
    from molexp.harness.schemas.bound_workflow import (
        BoundTask,
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )
    from molexp.harness.schemas.parameter import ParameterValue

    t1 = BoundTask(
        id="b1",
        ir_task_id="t1",
        capability_id="cap.x",
        package="pkg",
        callable="pkg.X",
        parameters={"n": ParameterValue(value=1, source="user_provided")},
        inputs={"n": "wf:n"},
        outputs={"out": "out.txt"},
    )
    return BoundWorkflow(
        id="bw-001",
        workflow_ir_id="wf-001",
        tasks=[t1],
        edges=[],
        execution_backend=backend,
        environment=ExecutionEnvironment(),
        resource_policy=ResourcePolicy(
            backend=backend,
            max_runtime_s=3600,
            max_memory_gb=8.0,
            denied_paths=["/", "~/.ssh"],
        ),
    )


def _empty_policy():
    """ApprovalPolicy with every require_for_* explicitly False."""
    from molexp.harness.schemas.policy import ApprovalPolicy

    return ApprovalPolicy(
        require_for_agent_inferred_scientific_parameters=False,
        require_for_full_execution=False,
        require_for_hpc_submission=False,
        require_for_large_resource_request=False,
        require_for_overwrite=False,
        require_for_final_report=False,
    )


# -------------------------------------------------------------- baseline


def test_baseline_no_triggers() -> None:
    """Clean workflow + all-False policy → zero requests."""
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    result = ApprovalPolicyEvaluator.evaluate(_empty_policy(), bw=_clean_bound_workflow())
    assert result == []


def test_bw_none_only_emits_via_helper() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator
    from molexp.harness.schemas.policy import ApprovalPolicy

    # All-True policy but bw is None → no auto-intents.
    result = ApprovalPolicyEvaluator.evaluate(ApprovalPolicy())
    assert result == []


# --------------------------------------------------------- hpc_submission


def test_hpc_submission_slurm() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_hpc_submission": True})
    bw = _clean_bound_workflow(backend="slurm")
    result = ApprovalPolicyEvaluator.evaluate(policy, bw=bw)
    intents = [r.intent for r in result]
    assert intents == ["hpc_submission"]


def test_hpc_submission_pbs_and_lsf() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_hpc_submission": True})
    for backend in ("pbs", "lsf"):
        bw = _clean_bound_workflow(backend=backend)
        result = ApprovalPolicyEvaluator.evaluate(policy, bw=bw)
        assert [r.intent for r in result] == ["hpc_submission"]


def test_hpc_submission_local_does_not_fire() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_hpc_submission": True})
    bw = _clean_bound_workflow(backend="local")
    assert ApprovalPolicyEvaluator.evaluate(policy, bw=bw) == []


def test_hpc_submission_policy_off_suppresses() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    bw = _clean_bound_workflow(backend="slurm")
    assert ApprovalPolicyEvaluator.evaluate(_empty_policy(), bw=bw) == []


# ----------------------------- agent_inferred_scientific_parameters


def test_agent_inferred_one_request_per_task() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator
    from molexp.harness.schemas.bound_workflow import (
        BoundTask,
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )
    from molexp.harness.schemas.parameter import ParameterValue

    t1 = BoundTask(
        id="b1",
        ir_task_id="t1",
        capability_id="cap.x",
        package="pkg",
        callable="pkg.X",
        parameters={
            "a": ParameterValue(value=1, source="agent_inferred"),
            "b": ParameterValue(value=2, source="agent_inferred"),
        },
        inputs={"a": "x", "b": "y"},
        outputs={"out": "out.txt"},
    )
    t2 = BoundTask(
        id="b2",
        ir_task_id="t2",
        capability_id="cap.y",
        package="pkg",
        callable="pkg.Y",
        parameters={"c": ParameterValue(value=3, source="agent_inferred")},
        inputs={"c": "z"},
        outputs={"out": "y.txt"},
    )
    t3 = BoundTask(
        id="b3",
        ir_task_id="t3",
        capability_id="cap.z",
        package="pkg",
        callable="pkg.Z",
        parameters={"d": ParameterValue(value=4, source="user_provided")},
        inputs={"d": "w"},
        outputs={"out": "z.txt"},
    )
    bw = BoundWorkflow(
        id="bw-001",
        workflow_ir_id="wf-001",
        tasks=[t1, t2, t3],
        edges=[],
        execution_backend="local",
        environment=ExecutionEnvironment(),
        resource_policy=ResourcePolicy(
            backend="local", max_runtime_s=3600, denied_paths=["/", "~/.ssh"]
        ),
    )

    policy = _empty_policy().model_copy(
        update={"require_for_agent_inferred_scientific_parameters": True}
    )
    result = ApprovalPolicyEvaluator.evaluate(policy, bw=bw)
    inferred = [r for r in result if r.intent == "agent_inferred_scientific_parameters"]
    assert len(inferred) == 2
    bt_ids = {r.metadata.get("bound_task_id") for r in inferred}
    assert bt_ids == {"b1", "b2"}


def test_agent_inferred_policy_off_suppresses() -> None:
    """Even with agent_inferred params, no request when policy flag off."""
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator
    from molexp.harness.schemas.bound_workflow import (
        BoundTask,
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )
    from molexp.harness.schemas.parameter import ParameterValue

    t1 = BoundTask(
        id="b1",
        ir_task_id="t1",
        capability_id="cap.x",
        package="pkg",
        callable="pkg.X",
        parameters={"a": ParameterValue(value=1, source="agent_inferred")},
        inputs={"a": "x"},
        outputs={"out": "out.txt"},
    )
    bw = BoundWorkflow(
        id="bw-001",
        workflow_ir_id="wf-001",
        tasks=[t1],
        edges=[],
        execution_backend="local",
        environment=ExecutionEnvironment(),
        resource_policy=ResourcePolicy(
            backend="local", max_runtime_s=3600, denied_paths=["/", "~/.ssh"]
        ),
    )
    assert ApprovalPolicyEvaluator.evaluate(_empty_policy(), bw=bw) == []


# ----------------------------------------- large_resource_request


def test_large_resource_runtime_just_above_24h() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_large_resource_request": True})
    bw = _clean_bound_workflow()
    bad = bw.resource_policy.model_copy(update={"max_runtime_s": 86401})
    bw = bw.model_copy(update={"resource_policy": bad})
    result = ApprovalPolicyEvaluator.evaluate(policy, bw=bw)
    assert [r.intent for r in result] == ["large_resource_request"]


def test_large_resource_runtime_at_24h_does_not_fire() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_large_resource_request": True})
    bw = _clean_bound_workflow()
    ok = bw.resource_policy.model_copy(update={"max_runtime_s": 86400})
    bw = bw.model_copy(update={"resource_policy": ok})
    assert ApprovalPolicyEvaluator.evaluate(policy, bw=bw) == []


def test_large_resource_memory_just_above_256() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_large_resource_request": True})
    bw = _clean_bound_workflow()
    bad = bw.resource_policy.model_copy(update={"max_memory_gb": 257.0})
    bw = bw.model_copy(update={"resource_policy": bad})
    result = ApprovalPolicyEvaluator.evaluate(policy, bw=bw)
    assert [r.intent for r in result] == ["large_resource_request"]


def test_large_resource_memory_at_256_does_not_fire() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_large_resource_request": True})
    bw = _clean_bound_workflow()
    ok = bw.resource_policy.model_copy(update={"max_memory_gb": 256.0})
    bw = bw.model_copy(update={"resource_policy": ok})
    assert ApprovalPolicyEvaluator.evaluate(policy, bw=bw) == []


def test_large_resource_none_memory_does_not_npe() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_large_resource_request": True})
    bw = _clean_bound_workflow()  # memory=8, runtime=3600
    # Set memory to None explicitly
    rp = bw.resource_policy.model_copy(update={"max_memory_gb": None})
    bw = bw.model_copy(update={"resource_policy": rp})
    assert ApprovalPolicyEvaluator.evaluate(policy, bw=bw) == []


def test_large_resource_both_breaches_one_request() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_large_resource_request": True})
    bw = _clean_bound_workflow()
    bad = bw.resource_policy.model_copy(update={"max_runtime_s": 90000, "max_memory_gb": 500.0})
    bw = bw.model_copy(update={"resource_policy": bad})
    result = ApprovalPolicyEvaluator.evaluate(policy, bw=bw)
    inferred = [r for r in result if r.intent == "large_resource_request"]
    assert len(inferred) == 1


# ----------------------------------------- full_execution


def test_full_execution_emits_with_bw_present() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_full_execution": True})
    result = ApprovalPolicyEvaluator.evaluate(policy, bw=_clean_bound_workflow())
    assert [r.intent for r in result] == ["full_execution"]


def test_full_execution_policy_off_suppresses() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    assert ApprovalPolicyEvaluator.evaluate(_empty_policy(), bw=_clean_bound_workflow()) == []


def test_full_execution_with_no_bw_skips() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator
    from molexp.harness.schemas.policy import ApprovalPolicy

    # Only full_execution=True; bw=None → no request.
    p = ApprovalPolicy(
        require_for_agent_inferred_scientific_parameters=False,
        require_for_full_execution=True,
        require_for_hpc_submission=False,
        require_for_large_resource_request=False,
        require_for_overwrite=False,
        require_for_final_report=False,
    )
    assert ApprovalPolicyEvaluator.evaluate(p) == []


# ----------------------------------------- overwrite


def test_overwrite_from_bw_review_flags() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_overwrite": True})
    bw = _clean_bound_workflow().model_copy(update={"review_flags": ["overwrite"]})
    result = ApprovalPolicyEvaluator.evaluate(policy, bw=bw)
    assert [r.intent for r in result] == ["overwrite"]


def test_overwrite_from_ir_task_review_flags() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.workflow_ir import TaskIR, WorkflowIR

    policy = _empty_policy().model_copy(update={"require_for_overwrite": True})
    bw = _clean_bound_workflow()  # no bw.review_flags
    t1 = TaskIR(
        id="t1",
        name="x",
        purpose="x",
        task_type="x",
        inputs={"n": ParameterValue(value=1, source="user_provided")},
        outputs={"out": "out.txt"},
        review_flags=["overwrite"],
    )
    ir = WorkflowIR(
        id="wf-001",
        name="wf",
        objective="x",
        inputs={"n": ParameterValue(value=1, source="user_provided")},
        tasks=[t1],
        edges=[],
        expected_outputs=[],
    )
    result = ApprovalPolicyEvaluator.evaluate(policy, bw=bw, ir=ir)
    assert [r.intent for r in result] == ["overwrite"]


def test_overwrite_neither_flag_suppresses() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator

    policy = _empty_policy().model_copy(update={"require_for_overwrite": True})
    bw = _clean_bound_workflow()
    assert ApprovalPolicyEvaluator.evaluate(policy, bw=bw) == []


def test_overwrite_both_flags_dedupe() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.workflow_ir import TaskIR, WorkflowIR

    policy = _empty_policy().model_copy(update={"require_for_overwrite": True})
    bw = _clean_bound_workflow().model_copy(update={"review_flags": ["overwrite"]})
    t1 = TaskIR(
        id="t1",
        name="x",
        purpose="x",
        task_type="x",
        inputs={"n": ParameterValue(value=1, source="user_provided")},
        outputs={"out": "out.txt"},
        review_flags=["overwrite"],
    )
    ir = WorkflowIR(
        id="wf-001",
        name="wf",
        objective="x",
        inputs={"n": ParameterValue(value=1, source="user_provided")},
        tasks=[t1],
        edges=[],
        expected_outputs=[],
    )
    result = ApprovalPolicyEvaluator.evaluate(policy, bw=bw, ir=ir)
    inferred = [r for r in result if r.intent == "overwrite"]
    assert len(inferred) == 1


# ----------------------------------------- final_report


def test_evaluate_never_emits_final_report() -> None:
    """final_report MUST NOT auto-trigger, even with require_for_final_report=True."""
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator
    from molexp.harness.schemas.policy import ApprovalPolicy

    policy = ApprovalPolicy(
        require_for_agent_inferred_scientific_parameters=True,
        require_for_full_execution=True,
        require_for_hpc_submission=True,
        require_for_large_resource_request=True,
        require_for_overwrite=True,
        require_for_final_report=True,
    )
    bw = _clean_bound_workflow().model_copy(update={"review_flags": ["overwrite"]})
    result = ApprovalPolicyEvaluator.evaluate(policy, bw=bw)
    assert "final_report" not in {r.intent for r in result}


def test_make_final_report_returns_request_when_policy_demands() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator
    from molexp.harness.schemas.policy import ApprovalPolicy

    req = ApprovalPolicyEvaluator.make_final_report_request(
        ApprovalPolicy(require_for_final_report=True)
    )
    assert req is not None
    assert req.intent == "final_report"
    assert req.id  # non-empty


def test_make_final_report_returns_none_when_policy_off() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator
    from molexp.harness.schemas.policy import ApprovalPolicy

    req = ApprovalPolicyEvaluator.make_final_report_request(
        ApprovalPolicy(require_for_final_report=False)
    )
    assert req is None


# ----------------------------------------- ordering


def test_result_ordering_is_deterministic() -> None:
    from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator
    from molexp.harness.schemas.bound_workflow import (
        BoundTask,
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )
    from molexp.harness.schemas.parameter import ParameterValue
    from molexp.harness.schemas.policy import ApprovalPolicy

    # Build a workflow that triggers all 5 auto-intents.
    t1 = BoundTask(
        id="b1",
        ir_task_id="t1",
        capability_id="cap.x",
        package="pkg",
        callable="pkg.X",
        parameters={"a": ParameterValue(value=1, source="agent_inferred")},
        inputs={"a": "x"},
        outputs={"out": "out.txt"},
    )
    bw = BoundWorkflow(
        id="bw-001",
        workflow_ir_id="wf-001",
        tasks=[t1],
        edges=[],
        execution_backend="slurm",
        environment=ExecutionEnvironment(),
        resource_policy=ResourcePolicy(
            backend="slurm",
            max_runtime_s=172800,
            denied_paths=["/", "~/.ssh"],
        ),
        review_flags=["overwrite"],
    )
    policy = ApprovalPolicy()  # all True

    r1 = ApprovalPolicyEvaluator.evaluate(policy, bw=bw)
    r2 = ApprovalPolicyEvaluator.evaluate(policy, bw=bw)
    intents_1 = [r.intent for r in r1]
    intents_2 = [r.intent for r in r2]
    assert intents_1 == intents_2
    # Documented order: hpc / full_execution / large_resource / overwrite / agent_inferred
    assert intents_1 == [
        "hpc_submission",
        "full_execution",
        "large_resource_request",
        "overwrite",
        "agent_inferred_scientific_parameters",
    ]


# ----------------------------------------- re-export


def test_evaluate_re_exported() -> None:
    from molexp.harness import (
        ApprovalPolicyEvaluator as top_eval,
    )
    from molexp.harness import (
        ApprovalPolicyEvaluator as top_final,
    )
    from molexp.harness.policy import (
        ApprovalPolicyEvaluator as pkg_eval,
    )
    from molexp.harness.policy import (
        ApprovalPolicyEvaluator as pkg_final,
    )

    assert top_eval is pkg_eval
    assert top_final is pkg_final
