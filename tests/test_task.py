from pathlib import Path
import pytest
import tempfile

import molexp as mx

class TestTask:

    @pytest.fixture
    def sample_task(self):
        return mx.Task(
            name="Sample Task",
            readme="This is a sample task",
            args=["--input", "data.txt"],
            kwargs={"output": "result.txt", "threads": 4}
        )

    def test_serialization_roundtrip(self, sample_task: mx.Task):
        # to_yaml + from_yaml string
        yaml_str = sample_task.to_yaml()
        recovered = mx.Task.from_yaml(yaml_str)
        assert recovered == sample_task

    def test_yaml_file_io(self, sample_task):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)/"task.yaml"
            sample_task.to_yaml(path)
            assert path.exists()

            loaded = mx.Task.from_yaml(path)
            assert loaded.name == sample_task.name
            assert loaded.readme == sample_task.readme
            assert loaded.args == sample_task.args
            assert loaded.kwargs == sample_task.kwargs

    def test_dict_serialization(self, sample_task):
        data = sample_task.model_dump()
        reconstructed = mx.Task(**data)
        assert reconstructed == sample_task

    def test_empty_args_kwargs(self):
        task = mx.Task(name="Empty", args=[], kwargs={})
        assert task.args == []
        assert task.kwargs == {}
