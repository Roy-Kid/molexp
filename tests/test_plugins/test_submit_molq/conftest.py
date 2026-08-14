"""Shared molq config fixture — write the format this molq actually reads."""

from __future__ import annotations

from pathlib import Path


def write_molq_demo_config(tmp_path: Path, monkeypatch) -> Path:
    """Point molq at *tmp_path* and write a ``demo`` local profile.

    Local checkouts still use ``config.toml``; published molq ≥0.8 uses YAML.
    Always write the filename :func:`molq.config.default_config_path` expects
    so ``Submitor.from_profile('demo')`` (no explicit path) works too.
    """
    monkeypatch.setenv("MOLCRAFTS_HOME", str(tmp_path))
    from molq.config import default_config_path

    dest = default_config_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    if dest.suffix == ".toml":
        dest.write_text(
            "[profiles.demo]\n"
            'scheduler = "local"\n'
            'cluster_name = "demo-local"\n'
            f'jobs_dir = "{jobs_dir}"\n'
        )
    else:
        dest.write_text(
            "profiles:\n"
            "  demo:\n"
            "    scheduler: local\n"
            "    cluster_name: demo-local\n"
            f'    jobs_dir: "{jobs_dir}"\n'
        )
    return dest
