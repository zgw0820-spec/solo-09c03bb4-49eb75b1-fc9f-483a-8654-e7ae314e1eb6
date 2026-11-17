"""SQLite schema and connection helpers.

One database file on a shared Docker volume, mounted into the API and both
worker containers.  Keeping jobs, checkpoints, leases and heartbeats in one
file means a checkpoint commit and the lease check that authorises it are a
single atomic ``BEGIN IMMEDIATE`` transaction.

SQLite runs in WAL mode with ``synchronous=FULL``: a committed batch survives a
process crash and a container restart.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    feature_dim     INTEGER NOT NULL,
    epochs          INTEGER NOT NULL,
    batch_size      INTEGER NOT NULL,
    record_count    INTEGER NOT NULL,
    total_batches   INTEGER NOT NULL,
    learning_rate   REAL NOT NULL,
    momentum        REAL NOT NULL,
    features_json   TEXT NOT NULL,
    labels_json     TEXT NOT NULL,
    weights_json    TEXT NOT NULL,
    velocity_json   TEXT NOT NULL,
    loss            REAL NOT NULL,
    completed_steps INTEGER NOT NULL DEFAULT 0,
    holdout_count   INTEGER NOT NULL DEFAULT 0,
    min_recall      REAL,
    max_false_positive_rate REAL,
    standardisation_json TEXT,
    holdout_features_json TEXT,
    holdout_labels_json TEXT,
    recall          REAL,
    false_positive_rate REAL,
    holdout_passed  INTEGER,
    evaluation_json TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    finished_at     REAL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    job_id        TEXT NOT NULL,
    step          INTEGER NOT NULL,
    weights_json  TEXT NOT NULL,
    velocity_json TEXT NOT NULL,
    loss          REAL NOT NULL,
    batch_loss    REAL NOT NULL,
    committed_at  REAL NOT NULL,
    worker_id     TEXT NOT NULL,
    lease_epoch   INTEGER NOT NULL,
    PRIMARY KEY (job_id, step),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS job_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT NOT NULL,
    event      TEXT NOT NULL,
    detail     TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS workers (
    worker_id    TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    started_at   REAL NOT NULL,
    heartbeat_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS leases (
    job_id       TEXT PRIMARY KEY,
    worker_id    TEXT NOT NULL,
    epoch        INTEGER NOT NULL,
    heartbeat_at REAL NOT NULL
);

-- Monotonic per-job lease counter.  Leases themselves are deleted on release,
-- but the counter is never reset, so a *new* lease can never reuse an epoch a
-- previous holder still holds a reference to.
CREATE TABLE IF NOT EXISTS lease_epochs (
    job_id TEXT PRIMARY KEY,
    epoch  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, id);
"""


# Columns added after the first release.  ``CREATE TABLE IF NOT EXISTS`` does
# not alter an existing table, so a database created by an older build is
# upgraded in place; every added column is nullable or has a default.
MIGRATIONS = {
    "jobs": (
        ("holdout_count", "INTEGER NOT NULL DEFAULT 0"),
        ("min_recall", "REAL"),
        ("max_false_positive_rate", "REAL"),
        ("standardisation_json", "TEXT"),
        ("holdout_features_json", "TEXT"),
        ("holdout_labels_json", "TEXT"),
        ("recall", "REAL"),
        ("false_positive_rate", "REAL"),
        ("holdout_passed", "INTEGER"),
        ("evaluation_json", "TEXT"),
    ),
}

# A job completed by the old build has no stored conclusion.  It cannot be
# evaluated retroactively (no held-out sample was submitted), so it is given
# an explicit "no verdict" conclusion.  That keeps the invariant that a
# completed job is never served without a conclusion, and keeps old rows
# queryable after the upgrade.
LEGACY_EVALUATION = {
    "evaluated": False,
    "threshold_met": None,
    "reason": "legacy_job_predates_holdout_evaluation",
    "prediction_threshold": 0.5,
    "holdout_records": 0,
    "positive_records": 0,
    "negative_records": 0,
    "recall": None,
    "false_positive_rate": None,
    "min_recall": None,
    "max_false_positive_rate": None,
    "true_positive": None,
    "false_positive": None,
    "true_negative": None,
    "false_negative": None,
}


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(path: Path | str) -> Iterator[sqlite3.Connection]:
    """A write transaction that takes the write lock immediately.

    ``BEGIN IMMEDIATE`` makes writers in separate containers queue on the busy
    timeout instead of failing on a deferred-to-exclusive lock upgrade.
    """
    conn = connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()


@contextmanager
def reader(path: Path | str) -> Iterator[sqlite3.Connection]:
    """A consistent read snapshot (WAL readers do not block writers)."""
    conn = connect(path)
    try:
        conn.execute("BEGIN")
        try:
            yield conn
        finally:
            conn.execute("COMMIT")
    finally:
        conn.close()


def init_db(path: Path | str) -> None:
    """Create the schema if missing and apply additive migrations.

    ``executescript`` commits implicitly, so it must not run inside our
    hand-managed transaction; concurrent starts serialise on SQLite's schema
    lock and every statement is idempotent.  Two containers may still race
    between "column missing" and ``ALTER TABLE``, so a duplicate-column error
    is treated as the other process having applied the same migration.
    """
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        for table, columns in MIGRATIONS.items():
            existing = {
                row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for name, declaration in columns:
                if name in existing:
                    continue
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
        conn.execute(
            "UPDATE jobs SET evaluation_json=? WHERE status='completed' "
            "AND evaluation_json IS NULL",
            (json.dumps(LEGACY_EVALUATION, allow_nan=False),),
        )
    finally:
        conn.close()
