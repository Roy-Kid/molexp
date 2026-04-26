"""Tests for run metrics API routes."""


class TestRunMetricsRoutes:
    def _prefix(self, project, experiment, run):
        return f"/api/projects/{project.id}/experiments/{experiment.id}/runs/{run.id}"

    def test_empty_metrics(self, client, project, experiment, run):
        resp = client.get(f"{self._prefix(project, experiment, run)}/metrics")

        assert resp.status_code == 200
        assert resp.json() == {
            "nextLine": 0,
            "records": [],
            "series": [],
            "parseErrors": 0,
        }

    def test_metrics_response(self, client, project, experiment, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.3, step=1)
            ctx.metrics.scalar("train/loss", 0.2, step=2)

        resp = client.get(f"{self._prefix(project, experiment, run)}/metrics")

        assert resp.status_code == 200
        data = resp.json()
        assert data["nextLine"] == 2
        assert [record["v"] for record in data["records"]] == [0.3, 0.2]
        assert data["series"] == [
            {
                "key": "train/loss",
                "type": "scalar",
                "count": 2,
                "latestStep": 2,
                "latestTimestamp": data["series"][0]["latestTimestamp"],
                "latestValue": 0.2,
            }
        ]

    def test_metrics_filters(self, client, project, experiment, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.3, step=1)
            ctx.metrics.text("note", "done", step=1)
            ctx.metrics.scalar("eval/acc", 0.8, step=1)

        resp = client.get(
            f"{self._prefix(project, experiment, run)}/metrics",
            params={"type": "scalar", "key": "eval/acc", "since_line": 1},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["nextLine"] == 3
        assert len(data["records"]) == 1
        assert data["records"][0]["k"] == "eval/acc"
