"""API contract tests against the real FastAPI app and a real SQLite file."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = tmp_path / "api.db"
    monkeypatch.setenv("TRAINING_DB_PATH", str(db))
    monkeypatch.setenv("TRAINING_LEASE_SECONDS", "0.6")
    monkeypatch.setenv("TRAINING_HEARTBEAT_INTERVAL", "0.1")

    import app.main as main_module

    monkeypatch.setattr(main_module, "settings", Settings.from_env())
    with TestClient(app) as test_client:
        yield test_client


def payload(records: int = 8, epochs: int = 2):
    rng = np.random.default_rng(1)
    return {
        "features": rng.normal(size=(records, 2)).tolist(),
        "labels": [i % 2 for i in range(records)],
        "epochs": epochs,
    }


def test_swagger_and_health(client):
    docs = client.get("/docs")
    assert docs.status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/jobs" in schema["paths"]
    assert "/jobs/{job_id}/checkpoints" in schema["paths"]
    health = client.get("/health").json()
    assert health["status"] == "ok"


def test_submit_returns_job_metadata(client):
    response = client.post("/jobs", json=payload(12, 3))
    assert response.status_code == 201
    job = response.json()
    assert job["batch_size"] == 8
    assert job["learning_rate"] == 0.1
    assert job["momentum"] == 0.9
    assert job["batches_per_epoch"] == 2
    assert job["total_steps"] == 6
    assert job["completed_steps"] == 0
    assert job["status"] == "queued"
    assert len(job["weights"]) == 3  # two standardised columns + bias


def test_query_progress_and_checkpoints(client):
    job = client.post("/jobs", json=payload(16, 2)).json()
    settings = Settings.from_env()

    from app import lease as lm
    from app import store

    lm.register_worker(settings.db_path, "w1")
    lease = lm.acquire(settings.db_path, job["job_id"], "w1", settings.lease_seconds)
    for _ in range(2):
        store.advance_job(settings.db_path, lease)

    current = client.get(f"/jobs/{job['job_id']}").json()
    assert current["completed_steps"] == 2
    assert current["total_steps"] == 4
    assert current["progress"] == pytest.approx(0.5)
    assert current["latest_checkpoint_step"] == 2
    assert current["final_weights"] is None

    checkpoints = client.get(f"/jobs/{job['job_id']}/checkpoints").json()
    assert [c["step"] for c in checkpoints] == [1, 2]
    assert checkpoints[0]["batch_start_index"] == 0
    assert checkpoints[1]["batch_start_index"] == 8
    assert all(c["worker_id"] == "w1" for c in checkpoints)
    assert all(len(c["weights"]) == 3 and len(c["momentum_velocity"]) == 3 for c in checkpoints)
    assert checkpoints[0]["training_epoch"] == 1 and checkpoints[0]["batch_in_epoch"] == 1
    assert checkpoints[1]["training_epoch"] == 1 and checkpoints[1]["batch_in_epoch"] == 2

    after = client.get(f"/jobs/{job['job_id']}/checkpoints", params={"after_step": 1}).json()
    assert [c["step"] for c in after] == [2]

    events = client.get(f"/jobs/{job['job_id']}/events").json()
    assert events[0]["event"] == "job_created"
    assert events[0]["detail"]["total_steps"] == 4

    workers = client.get("/workers").json()
    assert [w["worker_id"] for w in workers["workers"]] == ["w1"]
    assert workers["workers"][0]["alive"] is True
    assert workers["leases"][0]["lease_epoch"] == 1
    assert workers["leases"][0]["worker_id"] == "w1"


def test_validation_errors(client):
    bad = payload(7)
    assert client.post("/jobs", json=bad).status_code == 422
    assert client.post("/jobs", json=payload(8, 1)).status_code == 422
    assert client.post("/jobs", json=payload(8, 6)).status_code == 422
    mismatch = payload(8)
    mismatch["labels"] = mismatch["labels"][:-1]
    assert client.post("/jobs", json=mismatch).status_code == 422
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/jobs/does-not-exist/checkpoints").status_code == 404


def test_holdout_submission_and_validation(client):
    body = payload(12, 2)
    body.update(
        {
            "holdout_features": [[1.0, 2.0], [1.1, 2.1], [-1.0, -2.0], [-1.2, -2.2]],
            "holdout_labels": [1, 1, 0, 0],
            "min_recall": 0.5,
            "max_false_positive_rate": 0.25,
        }
    )
    response = client.post("/jobs", json=body)
    assert response.status_code == 201
    job = response.json()
    assert job["holdout"] == {
        "record_count": 4,
        "min_recall": 0.5,
        "max_false_positive_rate": 0.25,
    }
    assert job["evaluation"] is None
    assert job["recall"] is None and job["false_positive_rate"] is None

    schema = client.get("/openapi.json").json()
    props = schema["components"]["schemas"]["TrainingRequest"]["properties"]
    assert {"holdout_features", "holdout_labels", "min_recall", "max_false_positive_rate"} <= set(
        props
    )

    too_many = payload(8)
    too_many.update(
        {
            "holdout_features": [[0.1, 0.2]] * 17,
            "holdout_labels": [0, 1] * 8 + [0],
        }
    )
    assert client.post("/jobs", json=too_many).status_code == 422
    single_class = payload(8)
    single_class.update(
        {"holdout_features": [[0.1, 0.2], [0.3, 0.4]], "holdout_labels": [1, 1]}
    )
    assert client.post("/jobs", json=single_class).status_code == 422
    labels_without_features = payload(8)
    labels_without_features["holdout_labels"] = [0, 1]
    assert client.post("/jobs", json=labels_without_features).status_code == 422
    out_of_range = payload(8)
    out_of_range.update(
        {
            "holdout_features": [[0.1, 0.2], [0.3, 0.4]],
            "holdout_labels": [0, 1],
            "min_recall": 1.5,
        }
    )
    assert client.post("/jobs", json=out_of_range).status_code == 422
    wrong_width = payload(8)
    wrong_width.update({"holdout_features": [[0.1], [0.3]], "holdout_labels": [0, 1]})
    assert client.post("/jobs", json=wrong_width).status_code == 422
    legacy_ok = client.post("/jobs", json=payload(8))
    assert legacy_ok.status_code == 201
    assert legacy_ok.json()["holdout"] is None


def test_workers_endpoint_empty(client):
    data = client.get("/workers").json()
    assert "lease_seconds" in data
    assert data["workers"] == []
    assert data["leases"] == []
