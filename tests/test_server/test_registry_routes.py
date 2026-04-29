"""Tests for plugin and task-type registry routes."""


class TestTaskTypeRoutes:
    def test_list_includes_core_demo_slugs(self, client):
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        data = resp.json()
        slugs = {item["slug"] for item in data["task_types"]}
        # The core demo registry seeds these:
        assert {"core.constant", "core.add", "core.multiply"} <= slugs
        assert data["total"] == len(data["task_types"])

    def test_get_known_slug(self, client):
        resp = client.get("/api/tasks/core.add")
        assert resp.status_code == 200
        body = resp.json()
        assert body["slug"] == "core.add"
        assert "Sum" in body["description"]

    def test_get_unknown_slug_404(self, client):
        resp = client.get("/api/tasks/no.such.slug")
        assert resp.status_code == 404


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
