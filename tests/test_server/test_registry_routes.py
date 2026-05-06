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


# Plugin route shape is covered comprehensively by
# ``tests/test_server/test_plugins_route.py`` (per spec 07 split). Built-in
# ``core``/``metrics``/``molq`` plugins are statically imported on the
# frontend and do NOT appear in ``/api/plugins`` anymore — see
# ``test_plugins_route::test_builtin_ids_not_in_listing``.
