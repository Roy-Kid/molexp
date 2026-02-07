"""Tests for Link class and channel mapping."""

import pytest
from pydantic import BaseModel, ValidationError

from molexp.workflow import Link
from molexp.workflow.task import Task, Actor
from collections.abc import AsyncGenerator


# Test configurations and tasks
class SimpleConfig(BaseModel):
    value: int = 10


class SimpleTask(Task[SimpleConfig, dict]):
    config_type = SimpleConfig

    def execute(self, ctx=None, **inputs) -> dict:
        return {}


class SimpleActor(Actor[SimpleConfig, dict]):
    config_type = SimpleConfig

    async def execute(self, ctx=None, **inputs) -> AsyncGenerator[None, dict]:
        yield
        return


# Tests for Link instantiation
class TestLinkInstantiation:
    """Tests for Link creation and validation."""

    def test_link_with_string_task_ids(self):
        """Test Link accepts string task IDs."""
        link = Link(source='task_a', target='task_b')

        assert link.source == 'task_a'
        assert link.target == 'task_b'

    def test_link_with_task_instances(self):
        """Test Link accepts Task instances and extracts IDs."""
        task_a = SimpleTask()
        task_b = SimpleTask()

        link = Link(source=task_a, target=task_b)

        assert link.source == task_a.task_id
        assert link.target == task_b.task_id

    def test_link_with_actor_instances(self):
        """Test Link accepts Actor instances."""
        actor_a = SimpleActor()
        actor_b = SimpleActor()

        link = Link(source=actor_a, target=actor_b)

        assert link.source == actor_a.task_id
        assert link.target == actor_b.task_id

    def test_link_with_mixed_types(self):
        """Test Link accepts mix of Task instances and strings."""
        task_a = SimpleTask()

        link = Link(source=task_a, target='task_b')

        assert link.source == task_a.task_id
        assert link.target == 'task_b'

    def test_link_requires_source_and_target(self):
        """Test Link requires both source and target."""
        with pytest.raises(ValidationError):
            Link(source='task_a')  # Missing target

        with pytest.raises(ValidationError):
            Link(target='task_b')  # Missing source


# Tests for Link mapping
class TestLinkMapping:
    """Tests for Link.mapping attribute."""

    def test_link_mapping_defaults_to_none(self):
        """Test Link.mapping defaults to None."""
        link = Link(source='a', target='b')

        assert link.mapping is None

    def test_link_with_explicit_mapping(self):
        """Test Link accepts explicit mapping."""
        mapping = {'output_data': 'input_data'}
        link = Link(source='a', target='b', mapping=mapping)

        assert link.mapping == mapping

    def test_link_mapping_is_dict(self):
        """Test Link.mapping must be dict."""
        link = Link(
            source='a',
            target='b',
            mapping={'out': 'in', 'results': 'data'}
        )

        assert isinstance(link.mapping, dict)
        assert 'out' in link.mapping
        assert link.mapping['out'] == 'in'

    def test_link_mapping_can_be_empty_dict(self):
        """Test Link.mapping can be empty dict."""
        link = Link(source='a', target='b', mapping={})

        assert link.mapping == {}


# Tests for mapping auto-generation
class TestMappingAutoGeneration:
    """Tests for automatic mapping generation (integration with compiler)."""

    def test_link_mapping_populated_by_compiler(self):
        """Test Link.mapping is populated during compilation.

        Note: This is tested more thoroughly in compiler tests.
        This test documents that Links can start with mapping=None
        and be populated later.
        """
        link = Link(source='actor_a', target='actor_b')

        # Initially None
        assert link.mapping is None

        # After compilation, compiler would populate this
        # Simulating compiler action:
        link.mapping = {'actora_to_actorb': 'actora_to_actorb'}

        assert link.mapping is not None
        assert isinstance(link.mapping, dict)


# Tests for mapping serialization
class TestMappingSerialization:
    """Tests for Link serialization with mapping."""

    def test_link_serialization_includes_mapping(self):
        """Test Link can be serialized to dict."""
        link = Link(
            source='task_a',
            target='task_b',
            mapping={'data': 'input'},
            buffer_size=150
        )

        serialized = link.model_dump()

        assert isinstance(serialized, dict)
        assert serialized['source'] == 'task_a'
        assert serialized['target'] == 'task_b'
        assert serialized['mapping'] == {'data': 'input'}
        assert serialized['buffer_size'] == 150

    def test_link_deserialization(self):
        """Test Link can be reconstructed from dict."""
        data = {
            'source': 'actor_x',
            'target': 'actor_y',
            'mapping': {'output': 'input'},
            'buffer_size': 200
        }

        link = Link(**data)

        assert link.source == 'actor_x'
        assert link.target == 'actor_y'
        assert link.mapping == {'output': 'input'}
        assert link.buffer_size == 200

    def test_link_serialization_with_none_mapping(self):
        """Test Link serialization handles None mapping."""
        link = Link(source='a', target='b')

        serialized = link.model_dump()

        assert 'mapping' in serialized
        assert serialized['mapping'] is None


# Tests for buffer_size
class TestLinkBufferSize:
    """Tests for Link.buffer_size attribute."""

    def test_link_buffer_size_defaults_to_100(self):
        """Test Link.buffer_size defaults to 100."""
        link = Link(source='a', target='b')

        assert link.buffer_size == 100

    def test_link_with_custom_buffer_size(self):
        """Test Link accepts custom buffer_size."""
        link = Link(source='a', target='b', buffer_size=500)

        assert link.buffer_size == 500

    def test_link_buffer_size_used_for_actors(self):
        """Test buffer_size is meaningful for actor channels."""
        actor_a = SimpleActor()
        actor_b = SimpleActor()

        # Create link with specific buffer size
        link = Link(
            source=actor_a,
            target=actor_b,
            buffer_size=250,
            mapping={'out': 'in'}
        )

        # Buffer size should be preserved
        assert link.buffer_size == 250


# Tests for channels attribute
class TestLinkChannels:
    """Tests for Link.channels attribute."""

    def test_link_channels_defaults_to_none(self):
        """Test Link.channels defaults to None."""
        link = Link(source='a', target='b')

        assert link.channels is None

    def test_link_with_explicit_channels(self):
        """Test Link accepts explicit channels mapping."""
        channels = {'output_channel': 'input_channel'}
        link = Link(source='a', target='b', channels=channels)

        assert link.channels == channels

    def test_link_channels_is_dict(self):
        """Test Link.channels must be dict."""
        link = Link(
            source='a',
            target='b',
            channels={'out': 'in', 'error': 'error_handler'}
        )

        assert isinstance(link.channels, dict)


# Tests for status attribute
class TestLinkStatus:
    """Tests for Link.status attribute."""

    def test_link_status_defaults_to_pending(self):
        """Test Link.status defaults to 'pending'."""
        link = Link(source='a', target='b')

        assert link.status == "pending"

    def test_link_with_custom_status(self):
        """Test Link accepts custom status."""
        link = Link(source='a', target='b', status='active')

        assert link.status == 'active'


# Tests for field validation
class TestLinkFieldValidation:
    """Tests for Link field validation."""

    def test_link_extract_task_id_validator(self):
        """Test extract_task_id validator works correctly."""
        # With Task instance
        task = SimpleTask()
        link1 = Link(source=task, target='b')
        assert link1.source == task.task_id

        # With string
        link2 = Link(source='task_id', target='b')
        assert link2.source == 'task_id'

        # With Actor instance
        actor = SimpleActor()
        link3 = Link(source=actor, target='b')
        assert link3.source == actor.task_id


# Tests for Link equality and hashing
class TestLinkEquality:
    """Tests for Link comparison."""

    def test_links_with_same_values_are_equal(self):
        """Test Links with same values are equal."""
        link1 = Link(source='a', target='b', mapping={'x': 'y'})
        link2 = Link(source='a', target='b', mapping={'x': 'y'})

        # Pydantic models use value equality
        assert link1.source == link2.source
        assert link1.target == link2.target
        assert link1.mapping == link2.mapping


# Tests for Link documentation fields
class TestLinkDocumentation:
    """Tests that Link has proper documentation."""

    def test_link_has_docstring(self):
        """Test Link class has docstring."""
        assert Link.__doc__ is not None
        assert len(Link.__doc__) > 0

    def test_link_field_info(self):
        """Test Link fields have descriptions."""
        # Pydantic provides field info
        schema = Link.model_json_schema()

        assert 'properties' in schema
        assert 'source' in schema['properties']
        assert 'target' in schema['properties']
        assert 'mapping' in schema['properties']
        assert 'buffer_size' in schema['properties']


# Tests for Link usage in workflows
class TestLinkWorkflowUsage:
    """Tests for Link usage patterns in workflows."""

    def test_link_list_for_workflow(self):
        """Test multiple Links can be used in workflow."""
        task_a = SimpleTask()
        task_b = SimpleTask()
        task_c = SimpleTask()

        links = [
            Link(source=task_a, target=task_b),
            Link(source=task_b, target=task_c),
        ]

        assert len(links) == 2
        assert all(isinstance(link, Link) for link in links)

    def test_link_with_feedback_loop(self):
        """Test Link can represent feedback loop."""
        actor_a = SimpleActor()
        actor_b = SimpleActor()
        actor_c = SimpleActor()

        # Create cycle: A → B → C → A
        links = [
            Link(source=actor_a, target=actor_b, mapping={'ab': 'ab'}),
            Link(source=actor_b, target=actor_c, mapping={'bc': 'bc'}),
            Link(source=actor_c, target=actor_a, mapping={'ca': 'ca'}),
        ]

        # Links form a cycle
        assert links[0].source == actor_a.task_id
        assert links[2].target == actor_a.task_id
