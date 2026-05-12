from __future__ import annotations

from molexp.agent.capability_hints import CapabilityTriggerInput, TrustedNamespacePolicy


def test_required_direct_namespace_from_raw_user_input() -> None:
    policy = TrustedNamespacePolicy()
    hints = policy.extract(
        CapabilityTriggerInput(
            raw_user_input=(
                "You need to explicitly use molpy for building the polymer chain "
                "and use molpack to generate the initial configuration."
            )
        )
    )

    by_namespace = {hint.namespace: hint for hint in hints}
    assert by_namespace["molpy"].strength == "required"
    assert by_namespace["molpack"].strength == "required"


def test_preferred_alias_expansion_is_soft() -> None:
    policy = TrustedNamespacePolicy()
    hints = policy.extract(
        CapabilityTriggerInput(raw_user_input="Use the molcrafts toolchain wherever possible.")
    )

    assert {hint.namespace for hint in hints} == {"molpy", "molpack", "molexp"}
    assert {hint.strength for hint in hints} == {"preferred"}


def test_plain_project_alias_is_not_over_expanded() -> None:
    policy = TrustedNamespacePolicy()
    hints = policy.extract(
        CapabilityTriggerInput(project_digest="The plan may need Molcrafts symbols.")
    )

    assert hints == []


def test_no_hand_roll_constraint_tag() -> None:
    policy = TrustedNamespacePolicy()
    hints = policy.extract(
        CapabilityTriggerInput(
            raw_user_input=(
                "LAMMPS data must be written via molpy.io.write_lammps_data; "
                "do not hand-roll LAMMPS data."
            )
        )
    )

    tags = {tag for hint in hints for tag in hint.constraint_tags}
    assert "no_hand_rolled_lammps_data" in tags
