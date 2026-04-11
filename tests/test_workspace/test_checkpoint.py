"""Tests for checkpoint system."""


from molexp.workspace.checkpoint import Checkpoint, generate_checkpoint_id


class TestCheckpointState:
    def test_creation(self, checkpoint_state):
        assert checkpoint_state.ckpt_id == "ckpt_test123"
        assert checkpoint_state.run_id == "run-1"

    def test_defaults(self, checkpoint_state):
        assert checkpoint_state.version == "1.0"
        assert checkpoint_state.metadata == {}


class TestCheckpoint:
    def test_generate_id(self):
        cid = generate_checkpoint_id()
        assert cid.startswith("ckpt_")
        assert len(cid) > 5

    def test_save_creates_file(self, tmp_path, checkpoint_state):
        path = Checkpoint.save(tmp_path, checkpoint_state)
        assert path.exists()

    def test_save_creates_latest_symlink(self, tmp_path, checkpoint_state):
        Checkpoint.save(tmp_path, checkpoint_state)
        latest = tmp_path / "latest.json"
        assert latest.exists() or latest.is_symlink()

    def test_load_roundtrip(self, tmp_path, checkpoint_state):
        path = Checkpoint.save(tmp_path, checkpoint_state)
        loaded = Checkpoint.load(path)
        assert loaded.ckpt_id == checkpoint_state.ckpt_id
        assert loaded.context == checkpoint_state.context

    def test_get_latest_none_when_empty(self, tmp_path):
        assert Checkpoint.get_latest(tmp_path) is None

    def test_get_latest_returns_most_recent(self, tmp_path, checkpoint_state):
        Checkpoint.save(tmp_path, checkpoint_state)
        latest = Checkpoint.get_latest(tmp_path)
        assert latest is not None
        assert latest.ckpt_id == checkpoint_state.ckpt_id
