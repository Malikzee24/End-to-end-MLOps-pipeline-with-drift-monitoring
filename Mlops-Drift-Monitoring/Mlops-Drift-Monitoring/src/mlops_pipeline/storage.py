"""SQLite-backed store for predictions, delayed ground-truth labels and monitoring history.

SQLite keeps the project dependency-free and easy to run anywhere. The interface is small
(``PredictionStore``) so it can be swapped for Postgres / a feature store in production.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TS_FORMAT = "%Y-%m-%dT%H:%M:%S"


def to_iso(dt: datetime) -> str:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime(TS_FORMAT)


def from_iso(value: str) -> datetime:
    return datetime.strptime(value, TS_FORMAT)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    model_version TEXT NOT NULL,
    probability   REAL NOT NULL,
    prediction    INTEGER NOT NULL,
    features      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);

CREATE TABLE IF NOT EXISTS labels (
    prediction_id TEXT PRIMARY KEY,
    label         INTEGER NOT NULL,
    labeled_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monitoring_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status        TEXT NOT NULL,
    action        TEXT NOT NULL,
    n_current     INTEGER,
    n_labeled     INTEGER,
    share_drifted REAL,
    dataset_drift INTEGER,
    live_auc      REAL,
    baseline_auc  REAL,
    auc_drop      REAL,
    prediction_psi REAL,
    report_path   TEXT,
    details       TEXT
);

CREATE TABLE IF NOT EXISTS retrain_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    trigger        TEXT NOT NULL,
    old_version    TEXT,
    new_version    TEXT,
    promoted       INTEGER NOT NULL,
    champion_auc   REAL,
    challenger_auc REAL,
    details        TEXT
);
"""


class PredictionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ predictions & labels
    def log_predictions(self, records: list[dict]) -> None:
        """records: dicts with prediction_id, ts (datetime), model_version, probability, prediction, features."""
        rows = [
            (
                r["prediction_id"],
                to_iso(r["ts"]),
                str(r["model_version"]),
                float(r["probability"]),
                int(r["prediction"]),
                json.dumps(r["features"]),
            )
            for r in records
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?)",
                rows,
            )

    def add_labels(self, labels: list[tuple[str, int]], labeled_at: datetime | None = None) -> int:
        ts = to_iso(labeled_at or utcnow())
        with self._conn() as conn:
            cur = conn.executemany(
                "INSERT OR REPLACE INTO labels VALUES (?,?,?)",
                [(pid, int(lbl), ts) for pid, lbl in labels],
            )
            return cur.rowcount

    def get_predictions(self, since: datetime | None = None, until: datetime | None = None) -> pd.DataFrame:
        """Predictions in (since, until], joined with labels (NaN when label not yet available)."""
        sql = (
            "SELECT p.prediction_id, p.ts, p.model_version, p.probability, p.prediction, p.features, l.label "
            "FROM predictions p LEFT JOIN labels l ON l.prediction_id = p.prediction_id WHERE 1=1"
        )
        params: list = []
        if since is not None:
            sql += " AND p.ts > ?"
            params.append(to_iso(since))
        if until is not None:
            sql += " AND p.ts <= ?"
            params.append(to_iso(until))
        sql += " ORDER BY p.ts"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            return pd.DataFrame()
        meta = pd.DataFrame([dict(r) for r in rows])
        feats = pd.DataFrame([json.loads(f) for f in meta.pop("features")])
        out = pd.concat([meta.reset_index(drop=True), feats], axis=1)
        out["ts"] = pd.to_datetime(out["ts"])
        return out

    def count_predictions(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    # ------------------------------------------------------------------ monitoring history
    def save_monitoring_run(self, run: dict) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO monitoring_runs (ts, model_version, status, action, n_current, n_labeled, "
                "share_drifted, dataset_drift, live_auc, baseline_auc, auc_drop, prediction_psi, report_path, details) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    to_iso(run["ts"]),
                    str(run["model_version"]),
                    run["status"],
                    run["action"],
                    run.get("n_current"),
                    run.get("n_labeled"),
                    run.get("share_drifted"),
                    None if run.get("dataset_drift") is None else int(run["dataset_drift"]),
                    run.get("live_auc"),
                    run.get("baseline_auc"),
                    run.get("auc_drop"),
                    run.get("prediction_psi"),
                    run.get("report_path"),
                    json.dumps(run.get("details", {})),
                ),
            )
            return int(cur.lastrowid)

    def latest_monitoring_run(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM monitoring_runs ORDER BY id DESC LIMIT 1").fetchone()
        return self._decode(row)

    def monitoring_history(self) -> pd.DataFrame:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM monitoring_runs ORDER BY id").fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    # ------------------------------------------------------------------ retraining history
    def save_retrain_event(self, event: dict) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO retrain_events (ts, trigger, old_version, new_version, promoted, champion_auc, "
                "challenger_auc, details) VALUES (?,?,?,?,?,?,?,?)",
                (
                    to_iso(event["ts"]),
                    event["trigger"],
                    event.get("old_version"),
                    event.get("new_version"),
                    int(event["promoted"]),
                    event.get("champion_auc"),
                    event.get("challenger_auc"),
                    json.dumps(event.get("details", {})),
                ),
            )
            return int(cur.lastrowid)

    def last_retrain(self, promoted_only: bool = True) -> dict | None:
        sql = "SELECT * FROM retrain_events"
        if promoted_only:
            sql += " WHERE promoted = 1"
        sql += " ORDER BY id DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql).fetchone()
        return self._decode(row)

    def retrain_history(self) -> pd.DataFrame:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM retrain_events ORDER BY id").fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        if d.get("details"):
            d["details"] = json.loads(d["details"])
        return d
