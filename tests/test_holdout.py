"""Holdout-threshold tests: leak-free standardisation, final-batch verdict,
atomic persistence, and resumption after a failed final computation/write.
"""
from __future__ import annotations

import json
import sqlite3
import time

import numpy as np
import pytest
from app import lease as lm
from app import store
from app.config import Settings
from app.db import init_db, reader
from app.features import InputError, normalise
from app.lease import LeaseLost
from app.model import evaluate_holdout, sigmoid

HOLDOUT = {
    "holdout_features": [
        [0.9, 0.8, 4.9],
        [1.1, 0.9, 5.2],
        [-1.0, -0.9, -3.1],
        [-0.8, -0.7, -3.4],
    ],
    "holdout_labels": [1, 1, 0, 0],
    "min_recall": 0.5,
    "max_false_positive_rate": 0.5,
}


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "training.db",
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        poll_interval=0.01,
        batch_delay_seconds=0.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def settings(tmp_path):
    s = make_settings(tmp_path)
    s.validate()
    init_db(s.db_path)
    lm.register_worker(s.db_path, "w")
    return s


def sample(records: int = 10, seed: int = 0):
    """Linearly separable-ish data so a logistic model learns a real signal."""
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for index in range(records):
        label = index % 2
        centre = 1.5 if label else -1.5
        features.append(
            [float(round(centre + rng.normal(scale=0.5), 6)) for _ in range(3)]
        )
        labels.append(label)
    return features, labels


def lease_for(settings, job_id, worker_id="w"):
    lease = lm.acquire(settings.db_path, job_id, worker_id, settings.lease_seconds)
    assert lease is not None
    return lease


def drain(settings, lease, until=None):
    seen = []
    while True:
        result = store.advance_job(settings.db_path, lease)
        seen.append(result["step"])
        if result["status"] == "completed":
            return seen
        if until is not None and result["step"] >= until:
            return seen
        assert len(seen) < 1000


def test_holdout_validation(settings):
    features, labels = sample(10)
    with pytest.raises(InputError, match="between 2 and 16 holdout"):
        store.create_job(
            settings.db_path, features, labels, 2,
            holdout_features=[[0.1, 0.2, 0.3]], holdout_labels=[0],
        )
    with pytest.raises(InputError, match="between 2 and 16 holdout"):
        store.create_job(
            settings.db_path, features, labels, 2,
            holdout_features=[[0.1, 0.2, 0.3]] * 17, holdout_labels=[0, 1] * 8 + [0],
        )
    with pytest.raises(InputError, match="same 3 features"):
        store.create_job(
            settings.db_path, features, labels, 2,
            holdout_features=[[0.1, 0.2], [0.3, 0.4]], holdout_labels=[0, 1],
        )
    with pytest.raises(InputError, match="both healthy"):
        store.create_job(
            settings.db_path, features, labels, 2,
            holdout_features=[[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]], holdout_labels=[1, 1],
        )
    with pytest.raises(InputError, match="together"):
        store.create_job(
            settings.db_path, features, labels, 2,
            holdout_features=[[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]],
        )
    with pytest.raises(InputError, match="require holdout"):
        store.create_job(settings.db_path, features, labels, 2, min_recall=0.9)
    with pytest.raises(InputError, match="min_recall"):
        store.create_job(settings.db_path, features, labels, 2, **{**HOLDOUT, "min_recall": 1.5})
    with pytest.raises(InputError, match="max_false_positive_rate"):
        store.create_job(
            settings.db_path, features, labels, 2,
            **{**HOLDOUT, "max_false_positive_rate": -0.1},
        )


def test_holdout_uses_training_statistics_and_never_trains_on_holdout(settings):
    features, labels = sample(10, seed=3)
    holdout_features, holdout_labels = sample(6, seed=4)
    holdout_features = [[value * 10 + 5 for value in row] for row in holdout_features]

    with_holdout = store.create_job(
        settings.db_path, features, labels, 2,
        holdout_features=holdout_features, holdout_labels=holdout_labels,
        min_recall=0.0, max_false_positive_rate=1.0,
    )
    without = store.create_job(settings.db_path, features, labels, 2)
    assert with_holdout["record_count"] == without["record_count"] == 10
    assert with_holdout["total_steps"] == without["total_steps"]
    assert with_holdout["holdout"]["record_count"] == 6
    assert with_holdout["evaluation"] is None and with_holdout["recall"] is None

    with reader(settings.db_path) as conn:
        row = conn.execute(
            "SELECT standardisation_json, holdout_features_json FROM jobs WHERE job_id=?",
            (with_holdout["job_id"],),
        ).fetchone()
    stats = json.loads(row["standardisation_json"])
    design_holdout = np.asarray(json.loads(row["holdout_features_json"]), dtype=np.float64)
    expected_design, _ = normalise(features, labels)
    # Training design is untouched by the holdout attachment.
    assert np.allclose(expected_design[:, -1], 1.0)
    raw = np.asarray(features, dtype=np.float64)
    assert np.allclose(stats["mean"], raw.mean(axis=0), atol=1e-12)
    assert np.allclose(stats["std"], raw.std(axis=0), atol=1e-12)
    scaled = (np.asarray(holdout_features) - raw.mean(axis=0)) / raw.std(axis=0)
    assert np.allclose(design_holdout[:, :-1], scaled, atol=1e-12)
    assert np.allclose(design_holdout[:, -1], 1.0)

    # Same training data => identical weights whether or not holdout was attached.
    lease = lease_for(settings, with_holdout["job_id"])
    drain(settings, lease)
    lease = lease_for(settings, without["job_id"])
    drain(settings, lease)
    a = store.get_job(settings.db_path, with_holdout["job_id"])
    b = store.get_job(settings.db_path, without["job_id"])
    assert a["final_weights"] == b["final_weights"]
    assert a["loss"] == b["loss"]


def test_no_holdout_job_records_that_no_verdict_was_drawn(settings):
    job = store.create_job(settings.db_path, *sample(10), 2)
    lease = lease_for(settings, job["job_id"])
    drain(settings, lease)
    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    assert final["evaluation"]["evaluated"] is False
    assert final["evaluation"]["threshold_met"] is None
    assert final["evaluation"]["reason"] == "no_holdout_samples"
    assert final["recall"] is None and final["false_positive_rate"] is None
    events = store.list_events(settings.db_path, job["job_id"])
    assert events[-1]["event"] == "job_completed"
    assert events[-1]["detail"]["threshold_met"] is None


def test_final_batch_persists_metrics_and_verdict_with_progress(settings):
    features, labels = sample(12, seed=5)
    job = store.create_job(
        settings.db_path, features, labels, 2,
        holdout_features=HOLDOUT["holdout_features"],
        holdout_labels=HOLDOUT["holdout_labels"],
        min_recall=0.5, max_false_positive_rate=0.5,
    )
    lease = lease_for(settings, job["job_id"])
    partial = drain(settings, lease, until=job["total_steps"] - 1)
    assert partial[-1] == job["total_steps"] - 1
    before = store.get_job(settings.db_path, job["job_id"])
    assert before["status"] == "running"
    assert before["evaluation"] is None and before["threshold_met"] is None

    result = store.advance_job(settings.db_path, lease)
    assert result["status"] == "completed"
    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    assert final["completed_steps"] == final["total_steps"]
    assert final["progress"] == 1.0
    assert final["finished_at"] is not None

    # Independent metric computation on the same held-out data.
    raw = np.asarray(features, dtype=np.float64)
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    scaled = (np.asarray(HOLDOUT["holdout_features"]) - mean) / std
    design = np.concatenate([scaled, np.ones((scaled.shape[0], 1))], axis=1)
    probabilities = sigmoid(design @ np.asarray(final["final_weights"]))
    predicted = probabilities >= 0.5
    actual = np.asarray(HOLDOUT["holdout_labels"]) == 1
    recall = int(np.sum(predicted & actual)) / int(np.sum(actual))
    fpr = int(np.sum(predicted & ~actual)) / int(np.sum(~actual))
    expected = evaluate_holdout(
        np.asarray(final["final_weights"]), design,
        np.asarray(HOLDOUT["holdout_labels"], dtype=np.float64), 0.5, 0.5,
    )
    assert final["recall"] == pytest.approx(recall)
    assert final["false_positive_rate"] == pytest.approx(fpr)
    assert final["threshold_met"] is (recall >= 0.5 and fpr <= 0.5)
    assert final["evaluation"] == expected
    assert final["evaluation"]["prediction_threshold"] == 0.5

    # Persisted columns and evaluation JSON agree, and the verdict belongs to
    # the final batch only.
    with reader(settings.db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job["job_id"],)).fetchone()
    assert row["status"] == "completed"
    assert row["evaluation_json"] is not None
    assert row["holdout_passed"] == int(expected["threshold_met"])
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    assert checkpoints[-1]["step"] == final["total_steps"]
    assert checkpoints[-1]["weights"] == final["final_weights"]
    events = store.list_events(settings.db_path, job["job_id"])
    assert events[-1]["event"] == "job_completed"
    assert events[-1]["detail"]["recall"] == pytest.approx(recall)
    assert events[-1]["detail"]["threshold_met"] == expected["threshold_met"]


def test_negative_verdict_is_persisted(settings):
    job = store.create_job(
        settings.db_path, *sample(12, seed=7), 2,
        holdout_features=HOLDOUT["holdout_features"],
        holdout_labels=HOLDOUT["holdout_labels"],
        min_recall=0.0, max_false_positive_rate=0.0,
    )
    drain(settings, lease_for(settings, job["job_id"]))
    final = store.get_job(settings.db_path, job["job_id"])
    if final["false_positive_rate"] > 0.0:
        assert final["threshold_met"] is False
        assert final["evaluation"]["threshold_met"] is False
    else:  # pragma: no cover - pathologically perfect classifier
        assert final["threshold_met"] is True


def test_completed_without_conclusion_is_never_returned(settings):
    job = store.create_job(settings.db_path, *sample(8), 2)
    drain(settings, lease_for(settings, job["job_id"]))
    with store.transaction(settings.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET evaluation_json=NULL WHERE job_id=?", (job["job_id"],)
        )
    with pytest.raises(store.StateConflict, match="no stored evaluation"):
        store.get_job(settings.db_path, job["job_id"])
    with pytest.raises(store.StateConflict, match="no stored evaluation"):
        store.list_jobs(settings.db_path)


def test_migration_upgrades_a_pre_holdout_database(tmp_path):
    """A database created by the old build gains the holdout columns in place."""
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, status TEXT NOT NULL,
            feature_dim INTEGER NOT NULL, epochs INTEGER NOT NULL,
            batch_size INTEGER NOT NULL, record_count INTEGER NOT NULL,
            total_batches INTEGER NOT NULL, learning_rate REAL NOT NULL,
            momentum REAL NOT NULL, features_json TEXT NOT NULL,
            labels_json TEXT NOT NULL, weights_json TEXT NOT NULL,
            velocity_json TEXT NOT NULL, loss REAL NOT NULL,
            completed_steps INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL,
            finished_at REAL
        );
        """
    )
    conn.commit()
    conn.close()
    init_db(db)
    with reader(db) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    assert {
        "holdout_count",
        "min_recall",
        "max_false_positive_rate",
        "standardisation_json",
        "holdout_features_json",
        "holdout_labels_json",
        "recall",
        "false_positive_rate",
        "holdout_passed",
        "evaluation_json",
    } <= columns
    # init_db is idempotent.
    init_db(db)
    job = store.create_job(db, *sample(10), 2)
    assert job["holdout"] is None
    drain(Settings(db, 0.6, 0.1, 0.01, 0.0), lease_for(
        Settings(db, 0.6, 0.1, 0.01, 0.0), job["job_id"]
    ))
    assert store.get_job(db, job["job_id"])["status"] == "completed"


def test_legacy_completed_job_gains_an_explicit_no_verdict(tmp_path):
    """A completed row from the old build must stay queryable, with no verdict."""
    db = tmp_path / "legacy-completed.db"
    settings = Settings(db, 0.6, 0.1, 0.01, 0.0)
    init_db(db)
    job = store.create_job(db, *sample(8), 2)
    drain(settings, lease_for(settings, job["job_id"]))

    # Rename the raw completion into the old shape: no evaluation columns.
    with store.transaction(db) as conn:
        conn.execute(
            "UPDATE jobs SET evaluation_json=NULL, holdout_passed=NULL,"
            " recall=NULL, false_positive_rate=NULL WHERE job_id=?",
            (job["job_id"],),
        )
    with pytest.raises(store.StateConflict):
        store.get_job(db, job["job_id"])

    init_db(db)  # startup migration fills the conclusion
    repaired = store.get_job(db, job["job_id"])
    assert repaired["status"] == "completed"
    assert repaired["evaluation"]["evaluated"] is False
    assert repaired["evaluation"]["reason"] == "legacy_job_predates_holdout_evaluation"
    assert repaired["threshold_met"] is None
    assert repaired["final_weights"] is not None
    # Re-running init_db does not overwrite a real verdict.
    init_db(db)
    assert store.get_job(db, job["job_id"])["evaluation"] == repaired["evaluation"]


def _fill_until_before_final(settings, job_id, total_steps):
    lease = lease_for(settings, job_id, "w")
    drain(settings, lease, until=total_steps - 1)
    return lease


def test_failed_final_write_rolls_back_and_new_worker_resumes(settings):
    """A write failure inside the final batch leaves no partial verdict."""
    features, labels = sample(10, seed=11)
    job = store.create_job(
        settings.db_path, features, labels, 2,
        holdout_features=HOLDOUT["holdout_features"],
        holdout_labels=HOLDOUT["holdout_labels"],
        min_recall=0.5, max_false_positive_rate=0.5,
    )
    total = job["total_steps"]
    lease = _fill_until_before_final(settings, job["job_id"], total)

    # Inject a genuine write failure: a trigger aborts the final checkpoint
    # INSERT, so the whole final transaction must roll back.
    with store.transaction(settings.db_path) as conn:
        conn.execute(
            f"""
            CREATE TRIGGER fail_final_checkpoint BEFORE INSERT ON checkpoints
            WHEN NEW.job_id = '{job['job_id']}' AND NEW.step = {total}
            BEGIN
                SELECT RAISE(ABORT, 'simulated final checkpoint write failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="simulated final checkpoint"):
        store.advance_job(settings.db_path, lease)
    stuck = store.get_job(settings.db_path, job["job_id"])
    assert stuck["status"] == "running"
    assert stuck["completed_steps"] == total - 1
    assert stuck["evaluation"] is None
    assert stuck["threshold_met"] is None
    assert stuck["finished_at"] is None
    assert stuck["weights"] == [
        c["weights"]
        for c in store.list_checkpoints(settings.db_path, job["job_id"])
        if c["step"] == total - 1
    ][0]

    # Remove the injected failure; a different worker takes over and finishes.
    with store.transaction(settings.db_path) as conn:
        conn.execute("DROP TRIGGER fail_final_checkpoint")
    lm.release(settings.db_path, lease)
    lm.register_worker(settings.db_path, "w2")
    resumed = lease_for(settings, job["job_id"], "w2")
    assert resumed.epoch > lease.epoch
    result = store.advance_job(settings.db_path, resumed)
    assert result["status"] == "completed"
    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    assert final["completed_steps"] == total
    assert final["evaluation"]["evaluated"] is True
    assert [c["step"] for c in store.list_checkpoints(settings.db_path, job["job_id"])] == list(
        range(1, total + 1)
    )


def test_failed_final_evaluation_rolls_back_and_new_worker_resumes(settings, monkeypatch):
    """A failure while computing the final metrics is also resumable."""
    from app import store as store_module

    job = store.create_job(
        settings.db_path, *sample(14, seed=12), 2,
        holdout_features=HOLDOUT["holdout_features"],
        holdout_labels=HOLDOUT["holdout_labels"],
        min_recall=0.5, max_false_positive_rate=0.5,
    )
    lease = _fill_until_before_final(settings, job["job_id"], job["total_steps"])
    calls = {"n": 0}
    original = store_module.evaluate_holdout

    def explode(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("simulated metric computation failure")

    monkeypatch.setattr(store_module, "evaluate_holdout", explode)
    with pytest.raises(RuntimeError, match="simulated metric"):
        store.advance_job(settings.db_path, lease)
    assert calls["n"] == 1
    stuck = store.get_job(settings.db_path, job["job_id"])
    assert stuck["status"] == "running"
    assert stuck["completed_steps"] == job["total_steps"] - 1
    assert stuck["evaluation"] is None

    monkeypatch.setattr(store_module, "evaluate_holdout", original)
    lm.release(settings.db_path, lease)
    lm.register_worker(settings.db_path, "w2")
    resumed = lease_for(settings, job["job_id"], "w2")
    store.advance_job(settings.db_path, resumed)
    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    assert final["evaluation"]["evaluated"] is True


def test_interrupted_holdout_training_matches_uninterrupted(settings):
    features, labels = sample(19, seed=13)
    uninterrupted = store.create_job(
        settings.db_path, features, labels, 3,
        holdout_features=HOLDOUT["holdout_features"],
        holdout_labels=HOLDOUT["holdout_labels"],
        min_recall=0.5, max_false_positive_rate=0.5,
    )
    drain(settings, lease_for(settings, uninterrupted["job_id"], "w"))
    expected = store.get_job(settings.db_path, uninterrupted["job_id"])

    interrupted = store.create_job(
        settings.db_path, features, labels, 3,
        holdout_features=HOLDOUT["holdout_features"],
        holdout_labels=HOLDOUT["holdout_labels"],
        min_recall=0.5, max_false_positive_rate=0.5,
    )
    lease = lease_for(settings, interrupted["job_id"], "w")
    drain(settings, lease, until=4)
    lm.release(settings.db_path, lease)
    lm.register_worker(settings.db_path, "w2")
    resumed = lease_for(settings, interrupted["job_id"], "w2")
    drain(settings, resumed)
    actual = store.get_job(settings.db_path, interrupted["job_id"])

    assert actual["final_weights"] == expected["final_weights"]
    assert actual["loss"] == expected["loss"]
    assert actual["recall"] == expected["recall"]
    assert actual["false_positive_rate"] == expected["false_positive_rate"]
    assert actual["threshold_met"] == expected["threshold_met"]
    assert actual["evaluation"] == expected["evaluation"]


def test_late_worker_cannot_rewrite_verdict(settings):
    job = store.create_job(
        settings.db_path, *sample(8, seed=14), 2,
        holdout_features=HOLDOUT["holdout_features"],
        holdout_labels=HOLDOUT["holdout_labels"],
        min_recall=0.5, max_false_positive_rate=0.5,
    )
    lm.register_worker(settings.db_path, "old")
    lm.register_worker(settings.db_path, "new")
    old = lm.acquire(settings.db_path, job["job_id"], "old", settings.lease_seconds)
    store.advance_job(settings.db_path, old)
    time.sleep(settings.lease_seconds + 0.05)
    new = lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds)
    assert new is not None
    drain(settings, new)
    completed = store.get_job(settings.db_path, job["job_id"])
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, old)
    after = store.get_job(settings.db_path, job["job_id"])
    assert after["evaluation"] == completed["evaluation"]
    assert after["threshold_met"] == completed["threshold_met"]
    assert after["finished_at"] == completed["finished_at"]
