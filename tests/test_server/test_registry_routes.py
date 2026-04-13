"""Tests for plugin registry routes."""


class TestRegistryRoutes:
    def test_plugins_lists_core(self, client):
        resp = client.get("/api/plugins")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert any(plugin["id"] == "core" for plugin in data["plugins"])

    def test_plugins_lists_molq_manifest_when_supported(self, client, monkeypatch):
        monkeypatch.setattr(
            "molexp.server.routes.registry.discover_ui_plugins",
            lambda: [
                type(
                    "Descriptor",
                    (),
                    {
                        "id": "core",
                        "title": "Core Workspace UI",
                        "description": "Built-in Molexp workspace renderers and previews.",
                        "ui_module": "core",
                        "capabilities": ("workspace",),
                        "metadata": {},
                    },
                )(),
                type(
                    "Descriptor",
                    (),
                    {
                        "id": "molq",
                        "title": "Molq",
                        "description": "Scheduler-aware run viewers.",
                        "ui_module": "molq",
                        "capabilities": ("submit", "monitor"),
                        "metadata": {"schedulers": ["slurm"]},
                    },
                )(),
            ],
        )

        resp = client.get("/api/plugins")

        assert resp.status_code == 200
        data = resp.json()
        molq = next(plugin for plugin in data["plugins"] if plugin["id"] == "molq")
        assert molq["uiModule"] == "molq"
        assert molq["metadata"] == {"schedulers": ["slurm"]}
