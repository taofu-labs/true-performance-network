"""
SQLite-backed persistent storage for validator state.

All data stored in ~/.tpn/validator-storage/validator.db (POSIX) or
%APPDATA%/tpn/validator-storage/validator.db (Windows).

Leader-only tables (scoring_runs, scoring_results, reveals, precheck_outcomes)
record the full history of every scoring run — permanent, append-only.
Both leader and follower use scored_competitions/banned_hotkeys/weights_history.

WAL mode lets the read API (leader.api) and the scoring loop read/write the
same file concurrently from one process without lock contention.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from common.models.submission import MinerSubmission
    from common.models.competition import CompetitionSpec
    from validator.scorer import ScoringOutcome


def tpn_home() -> Path:
    """Platform-aware base dir: ~/.tpn on POSIX, %APPDATA%/tpn on Windows."""
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / "tpn"
    return Path.home() / ".tpn"


def validator_storage_dir() -> Path:
    """Returns ~/.tpn/validator-storage/, creating it if necessary."""
    d = tpn_home() / "validator-storage"
    d.mkdir(parents=True, exist_ok=True)
    return d


def validator_db_path() -> Path:
    return validator_storage_dir() / "validator.db"


def clear_validator_db() -> None:
    """Delete the validator.db file (and any WAL/SHM sidecars)."""
    base = validator_db_path()
    for p in (base, base.with_name(base.name + "-wal"), base.with_name(base.name + "-shm")):
        p.unlink(missing_ok=True)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS scoring_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id TEXT NOT NULL,
    scored_at REAL NOT NULL,
    block INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS scoring_results (
    run_id INTEGER NOT NULL REFERENCES scoring_runs(id),
    competition_id TEXT NOT NULL,
    hotkey TEXT NOT NULL,
    passed_floors INTEGER NOT NULL,
    disqualified INTEGER NOT NULL,
    disqualification_reason TEXT,
    actual_scores TEXT NOT NULL,
    final_score REAL NOT NULL,
    max_memory_kb INTEGER NOT NULL,
    lying_detected INTEGER NOT NULL,
    eval_backend TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reveals (
    run_id INTEGER NOT NULL REFERENCES scoring_runs(id),
    hotkey TEXT NOT NULL,
    submission_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS precheck_outcomes (
    run_id INTEGER NOT NULL REFERENCES scoring_runs(id),
    hotkey TEXT NOT NULL,
    kind TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS weights_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id TEXT NOT NULL,
    set_at REAL NOT NULL,
    weights_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scored_competitions (
    competition_id TEXT PRIMARY KEY,
    scored_at REAL NOT NULL,
    reveal_attempts INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS banned_hotkeys (
    hotkey TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    banned_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS competitions (
    id TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scoring_runs_competition ON scoring_runs(competition_id);
CREATE INDEX IF NOT EXISTS idx_scoring_results_run ON scoring_results(run_id);
CREATE INDEX IF NOT EXISTS idx_reveals_run ON reveals(run_id);
CREATE INDEX IF NOT EXISTS idx_precheck_outcomes_run ON precheck_outcomes(run_id);
"""


_connections: Dict[str, sqlite3.Connection] = {}


def init_db(path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Open (creating if needed) the validator SQLite DB with WAL mode.

    Returns the same Connection object for repeated calls with the same path
    within a process, so callers that open "the" validator DB independently
    share one handle rather than two.
    """
    db_path = path or validator_db_path()
    key = str(db_path)
    if key in _connections:
        return _connections[key]

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    _connections[key] = conn
    return conn


# ---------------------------------------------------------------------------
# Leader-only writes
# ---------------------------------------------------------------------------

def record_scoring_run(
    conn: sqlite3.Connection,
    competition_id: str,
    block: int,
    outcomes: List["ScoringOutcome"],
    reveals: Dict[str, "MinerSubmission"],
) -> int:
    """Persist one full scoring run: results, precheck outcomes, raw reveals. Returns run_id."""
    now = time.time()
    cur = conn.execute(
        "INSERT INTO scoring_runs (competition_id, scored_at, block) VALUES (?, ?, ?)",
        (competition_id, now, block),
    )
    run_id = cur.lastrowid

    for outcome in outcomes:
        r = outcome.result
        conn.execute(
            """INSERT INTO scoring_results
               (run_id, competition_id, hotkey, passed_floors, disqualified, disqualification_reason,
                actual_scores, final_score, max_memory_kb, lying_detected, eval_backend)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, competition_id, r.hotkey, int(r.passed_floors), int(r.disqualified),
                r.disqualification_reason, json.dumps(r.actual_scores), r.final_score,
                r.max_memory_kb, int(r.lying_detected), r.eval_backend,
            ),
        )
        conn.execute(
            "INSERT INTO precheck_outcomes (run_id, hotkey, kind, reason) VALUES (?, ?, ?, ?)",
            (run_id, r.hotkey, outcome.kind.name, outcome.reason),
        )

    for hotkey, submission in reveals.items():
        conn.execute(
            "INSERT INTO reveals (run_id, hotkey, submission_json) VALUES (?, ?, ?)",
            (run_id, hotkey, submission.model_dump_json()),
        )

    conn.commit()
    return run_id


def record_weights(conn: sqlite3.Connection, competition_id: str, hotkey_weights: Dict[str, float]) -> None:
    conn.execute(
        "INSERT INTO weights_history (competition_id, set_at, weights_json) VALUES (?, ?, ?)",
        (competition_id, time.time(), json.dumps(hotkey_weights)),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Leader-only reads (serve the API)
# ---------------------------------------------------------------------------

def scoring_results_since(conn: sqlite3.Connection, competition_id: str, after_run_id: int = 0) -> List[dict]:
    """Runs for a competition with id > after_run_id, each with its results embedded. For follower polling."""
    runs = conn.execute(
        "SELECT * FROM scoring_runs WHERE competition_id = ? AND id > ? ORDER BY id ASC",
        (competition_id, after_run_id),
    ).fetchall()

    out = []
    for run in runs:
        results = conn.execute(
            "SELECT * FROM scoring_results WHERE run_id = ?", (run["id"],)
        ).fetchall()
        out.append({
            "run_id": run["id"],
            "competition_id": run["competition_id"],
            "block": run["block"],
            "scored_at": run["scored_at"],
            "results": [_result_row_to_dict(r) for r in results],
        })
    return out


def weights_history_for_competition(conn: sqlite3.Connection, competition_id: str) -> List[dict]:
    rows = conn.execute(
        "SELECT * FROM weights_history WHERE competition_id = ? ORDER BY id ASC",
        (competition_id,),
    ).fetchall()
    return [{"set_at": w["set_at"], "weights": json.loads(w["weights_json"])} for w in rows]


def latest_weights_for_competition(conn: sqlite3.Connection, competition_id: str) -> Optional[Dict[str, float]]:
    """Most recently recorded per-hotkey weights for a competition, or None if never scored."""
    row = conn.execute(
        "SELECT weights_json FROM weights_history WHERE competition_id = ? ORDER BY id DESC LIMIT 1",
        (competition_id,),
    ).fetchone()
    return json.loads(row["weights_json"]) if row else None


def full_state_for_competition(conn: sqlite3.Connection, competition_id: str) -> dict:
    """Reveals + precheck outcomes + scoring results + weights history for one competition. Dashboard use."""
    runs = conn.execute(
        "SELECT * FROM scoring_runs WHERE competition_id = ? ORDER BY id ASC",
        (competition_id,),
    ).fetchall()

    run_details = []
    for run in runs:
        run_id = run["id"]
        results = conn.execute("SELECT * FROM scoring_results WHERE run_id = ?", (run_id,)).fetchall()
        precheck = conn.execute("SELECT * FROM precheck_outcomes WHERE run_id = ?", (run_id,)).fetchall()
        reveals = conn.execute("SELECT * FROM reveals WHERE run_id = ?", (run_id,)).fetchall()
        run_details.append({
            "run_id": run_id,
            "block": run["block"],
            "scored_at": run["scored_at"],
            "results": [_result_row_to_dict(r) for r in results],
            "precheck_outcomes": [
                {"hotkey": p["hotkey"], "kind": p["kind"], "reason": p["reason"]} for p in precheck
            ],
            "reveals": {
                r["hotkey"]: json.loads(r["submission_json"]) for r in reveals
            },
        })

    return {
        "competition_id": competition_id,
        "runs": run_details,
        "weights_history": weights_history_for_competition(conn, competition_id),
    }


def _result_row_to_dict(r: sqlite3.Row) -> dict:
    return {
        "hotkey": r["hotkey"],
        "competition_id": r["competition_id"],
        "passed_floors": bool(r["passed_floors"]),
        "disqualified": bool(r["disqualified"]),
        "disqualification_reason": r["disqualification_reason"],
        "actual_scores": json.loads(r["actual_scores"]),
        "final_score": r["final_score"],
        "max_memory_kb": r["max_memory_kb"],
        "lying_detected": bool(r["lying_detected"]),
        "eval_backend": r["eval_backend"],
    }


# ---------------------------------------------------------------------------
# Both modes — "already scored this competition" bookkeeping
# ---------------------------------------------------------------------------

MAX_REVEAL_ATTEMPTS = 3  # 1 initial + 2 retries


def mark_scored(conn: sqlite3.Connection, competition_id: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO scored_competitions (competition_id, scored_at) VALUES (?, ?)",
        (competition_id, time.time()),
    )
    conn.commit()


def is_scored(conn: sqlite3.Connection, competition_id: str) -> bool:
    row = conn.execute(
        "SELECT scored_at FROM scored_competitions WHERE competition_id = ?", (competition_id,)
    ).fetchone()
    return row is not None and row["scored_at"] > 0


def bump_reveal_attempts(conn: sqlite3.Connection, competition_id: str) -> int:
    """Increment and return the retry-attempt count for competition_id (row created if absent)."""
    conn.execute(
        "INSERT INTO scored_competitions (competition_id, scored_at, reveal_attempts) "
        "VALUES (?, 0, 1) "
        "ON CONFLICT(competition_id) DO UPDATE SET reveal_attempts = reveal_attempts + 1",
        (competition_id,),
    )
    conn.commit()
    row = conn.execute(
        "SELECT reveal_attempts FROM scored_competitions WHERE competition_id = ?", (competition_id,)
    ).fetchone()
    return row["reveal_attempts"]


# ---------------------------------------------------------------------------
# Leader-only — competition specs (source of truth for /v1/competitions)
# ---------------------------------------------------------------------------

def upsert_competition(conn: sqlite3.Connection, spec: "CompetitionSpec") -> None:
    """Insert or replace a competition spec, keeping the original created_at on update."""
    now = time.time()
    spec_json = spec.model_dump_json()
    conn.execute(
        """INSERT INTO competitions (id, spec_json, created_at, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET spec_json = excluded.spec_json, updated_at = excluded.updated_at""",
        (spec.id, spec_json, now, now),
    )
    conn.commit()


def get_competition(conn: sqlite3.Connection, competition_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT spec_json FROM competitions WHERE id = ?", (competition_id,)
    ).fetchone()
    return json.loads(row["spec_json"]) if row else None


def list_competitions(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute("SELECT spec_json FROM competitions ORDER BY id ASC").fetchall()
    return [json.loads(r["spec_json"]) for r in rows]


# ---------------------------------------------------------------------------
# Both modes — bans
# ---------------------------------------------------------------------------

def ban(conn: sqlite3.Connection, hotkey: str, reason: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO banned_hotkeys (hotkey, reason, banned_at) VALUES (?, ?, ?)",
        (hotkey, reason, time.time()),
    )
    conn.commit()


def is_banned(conn: sqlite3.Connection, hotkey: str) -> bool:
    row = conn.execute("SELECT 1 FROM banned_hotkeys WHERE hotkey = ?", (hotkey,)).fetchone()
    return row is not None
