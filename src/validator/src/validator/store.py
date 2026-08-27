"""
SQLite-backed persistent storage for validator state.

All data stored in ~/.tpn/validator-storage/validator.db (POSIX) or
%APPDATA%/tpn/validator-storage/validator.db (Windows).
"""
from __future__ import annotations

import functools
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


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
CREATE TABLE IF NOT EXISTS scored_competitions (
    competition_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL DEFAULT 'stage1_ranking',
    stage1_attempts INTEGER NOT NULL DEFAULT 0,
    scored_at REAL,
    status TEXT NOT NULL DEFAULT 'scoring'
);

CREATE TABLE IF NOT EXISTS revealed_candidates (
    competition_id TEXT NOT NULL,
    hotkey TEXT NOT NULL,
    rank INTEGER NOT NULL,
    submission_json TEXT NOT NULL,
    reveal_block INTEGER NOT NULL,
    status TEXT NOT NULL,
    failure_reason TEXT,
    gguf_file TEXT,
    measured_memory_kb INTEGER,
    updated_at REAL NOT NULL,
    PRIMARY KEY (competition_id, hotkey)
);

CREATE TABLE IF NOT EXISTS benchmark_results (
    competition_id TEXT NOT NULL,
    hotkey TEXT NOT NULL,
    benchmark_name TEXT NOT NULL,
    score REAL,
    coordinator_run_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'submitted',
    repository TEXT NOT NULL,
    revision TEXT NOT NULL,
    submitted_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    phase TEXT,
    percent_complete REAL,
    last_message TEXT,
    PRIMARY KEY (competition_id, hotkey, benchmark_name)
);

CREATE TABLE IF NOT EXISTS scoring_results (
    competition_id TEXT NOT NULL,
    hotkey TEXT NOT NULL,
    final_score REAL NOT NULL,
    max_memory_kb INTEGER NOT NULL,
    finalized_at REAL NOT NULL,
    PRIMARY KEY (competition_id, hotkey)
);

CREATE TABLE IF NOT EXISTS weights_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id TEXT NOT NULL,
    set_at REAL NOT NULL,
    weights_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    competition_id TEXT NOT NULL,
    hotkey TEXT NOT NULL,
    benchmark_name TEXT NOT NULL,
    repository TEXT NOT NULL,
    revision TEXT NOT NULL,
    coordinator_run_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'submitted',
    submitted_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    phase TEXT,
    percent_complete REAL,
    last_message TEXT,
    PRIMARY KEY (competition_id, hotkey, benchmark_name)
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

CREATE INDEX IF NOT EXISTS idx_revealed_candidates_competition ON revealed_candidates(competition_id);
CREATE INDEX IF NOT EXISTS idx_revealed_candidates_status ON revealed_candidates(competition_id, status);
CREATE INDEX IF NOT EXISTS idx_benchmark_results_competition ON benchmark_results(competition_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_results_status ON benchmark_results(competition_id, status);
CREATE INDEX IF NOT EXISTS idx_scoring_results_competition ON scoring_results(competition_id);
"""


_connections: Dict[str, sqlite3.Connection] = {}

# One process-wide lock guarding the shared Connection. The validator runs its
# sync store calls from two places: the asyncio event loop (leader/weight loops
# and the aiohttp API handlers) and the one asyncio.to_thread worker that
# run_stage_2 hands blocking work to. Python's sqlite3 is built with
# threadsafety=1 — module-level serialization only — so a Connection opened
# check_same_thread=False is NOT safe to use from two threads concurrently.
# Every public store function below is wrapped in @_locked.
#
# Note this makes each call safe, not each pair of calls: commit() is
# connection-global, so a commit on one thread also lands whatever the other
# thread had open. Nothing in the current call graph runs two writers at once
# (run_stage_2 awaits its to_thread calls one at a time), and the admin API's
# only writer that could interleave is reset-scoring.
_LOCK = threading.RLock()


def _locked(fn):
    """Serialize a store function against the shared Connection."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _LOCK:
            return fn(*args, **kwargs)
    return wrapper


@_locked
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
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    conn.commit()
    _migrate_scored_competitions_stage_columns(conn)
    _connections[key] = conn
    return conn


def _migrate_scored_competitions_stage_columns(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(scored_competitions)")}
    if "stage" not in existing:
        conn.execute("ALTER TABLE scored_competitions ADD COLUMN stage TEXT NOT NULL DEFAULT 'finalized'")
    if "stage1_attempts" not in existing:
        conn.execute("ALTER TABLE scored_competitions ADD COLUMN stage1_attempts INTEGER NOT NULL DEFAULT 0")
    conn.execute("UPDATE benchmark_results SET status = 'submitted' WHERE status = 'pending-resume'")
    conn.commit()


STAGE1_MAX_ATTEMPTS = 3


@_locked
def get_stage(conn: sqlite3.Connection, competition_id: str) -> str:
    row = conn.execute(
        "SELECT stage FROM scored_competitions WHERE competition_id = ?", (competition_id,)
    ).fetchone()
    return row["stage"] if row else "stage1_ranking"


@_locked
def bump_stage1_attempts(conn: sqlite3.Connection, competition_id: str) -> int:
    conn.execute(
        "INSERT INTO scored_competitions (competition_id, stage, stage1_attempts) "
        "VALUES (?, 'stage1_ranking', 1) "
        "ON CONFLICT(competition_id) DO UPDATE SET stage1_attempts = stage1_attempts + 1",
        (competition_id,),
    )
    conn.commit()
    row = conn.execute(
        "SELECT stage1_attempts FROM scored_competitions WHERE competition_id = ?", (competition_id,)
    ).fetchone()
    return row["stage1_attempts"]


@_locked
def set_stage(conn: sqlite3.Connection, competition_id: str, stage: str) -> None:
    conn.execute(
        "INSERT INTO scored_competitions (competition_id, stage) VALUES (?, ?) "
        "ON CONFLICT(competition_id) DO UPDATE SET stage = excluded.stage",
        (competition_id, stage),
    )
    conn.commit()


@_locked
def mark_scored(conn: sqlite3.Connection, competition_id: str, status: str = "scored") -> None:
    stage = "finalized" if status == "scored" else status
    conn.execute(
        "INSERT INTO scored_competitions (competition_id, scored_at, status, stage) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(competition_id) DO UPDATE SET scored_at = excluded.scored_at, "
        "status = excluded.status, stage = excluded.stage",
        (competition_id, time.time(), status, stage),
    )
    conn.commit()


@_locked
def is_scored(conn: sqlite3.Connection, competition_id: str) -> bool:
    row = conn.execute(
        "SELECT scored_at FROM scored_competitions WHERE competition_id = ?", (competition_id,)
    ).fetchone()
    return row is not None and row["scored_at"] is not None and row["scored_at"] > 0


@_locked
def scored_status(conn: sqlite3.Connection, competition_id: str) -> Optional[str]:
    row = conn.execute(
        "SELECT status FROM scored_competitions WHERE competition_id = ?", (competition_id,)
    ).fetchone()
    return row["status"] if row else None


@_locked
def reset_competition_scoring(conn: sqlite3.Connection, competition_id: str) -> dict:
    """Clear scoring state for one competition so it re-runs from stage 1.

    Keeps bans and weights history. Returns per-table deleted row counts.
    Callers must refuse anything past stage 1 that did not fail.
    """
    deleted = {}
    for table in ("revealed_candidates", "benchmark_results", "benchmark_runs", "scoring_results"):
        cur = conn.execute(f"DELETE FROM {table} WHERE competition_id = ?", (competition_id,))
        deleted[table] = cur.rowcount
    cur = conn.execute("DELETE FROM scored_competitions WHERE competition_id = ?", (competition_id,))
    deleted["scored_competitions"] = cur.rowcount
    conn.commit()
    return deleted



@_locked
def insert_revealed_candidate(
    conn: sqlite3.Connection,
    competition_id: str,
    hotkey: str,
    rank: int,
    submission_json: str,
    reveal_block: int,
    status: str,
    failure_reason: Optional[str] = None,
) -> None:
    conn.execute(
        """INSERT INTO revealed_candidates
           (competition_id, hotkey, rank, submission_json, reveal_block, status, failure_reason, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (competition_id, hotkey, rank, submission_json, reveal_block, status, failure_reason, time.time()),
    )
    conn.commit()


@_locked
def insert_revealed_candidates(
    conn: sqlite3.Connection, competition_id: str, candidates: List[dict]
) -> None:
    now = time.time()
    with conn:
        conn.executemany(
            """INSERT INTO revealed_candidates
               (competition_id, hotkey, rank, submission_json, reveal_block, status, failure_reason, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (competition_id, c["hotkey"], c["rank"], c["submission_json"],
                 c["reveal_block"], c["status"], c.get("failure_reason"), now)
                for c in candidates
            ],
        )


@_locked
def set_candidate_status(
    conn: sqlite3.Connection,
    competition_id: str,
    hotkey: str,
    status: str,
    failure_reason: Optional[str] = None,
) -> None:
    conn.execute(
        """UPDATE revealed_candidates SET status = ?, failure_reason = ?, updated_at = ?
           WHERE competition_id = ? AND hotkey = ?""",
        (status, failure_reason, time.time(), competition_id, hotkey),
    )
    conn.commit()


@_locked
def reset_stale_candidate_statuses(conn: sqlite3.Connection, competition_id: str) -> int:
    """Recover candidates left mid-precheck by a process that died before
    writing a terminal status. Safe only at startup — nothing is genuinely
    in flight the moment a fresh process boots. Returns count reset."""
    cur = conn.execute(
        """UPDATE revealed_candidates SET status = 'standby', updated_at = ?
           WHERE competition_id = ? AND status = 'prechecking'""",
        (time.time(), competition_id),
    )
    conn.commit()
    return cur.rowcount


@_locked
def mark_precheck_passed(
    conn: sqlite3.Connection,
    competition_id: str,
    hotkey: str,
    gguf_file: str,
    measured_memory_kb: int,
) -> None:
    conn.execute(
        """UPDATE revealed_candidates
           SET status = 'queued', gguf_file = ?, measured_memory_kb = ?, updated_at = ?
           WHERE competition_id = ? AND hotkey = ?""",
        (gguf_file, measured_memory_kb, time.time(), competition_id, hotkey),
    )
    conn.commit()


@_locked
def get_candidate(conn: sqlite3.Connection, competition_id: str, hotkey: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM revealed_candidates WHERE competition_id = ? AND hotkey = ?",
        (competition_id, hotkey),
    ).fetchone()
    return dict(row) if row else None


@_locked
def candidates_by_status(
    conn: sqlite3.Connection, competition_id: str, statuses: tuple, order_by_rank: bool = True
) -> List[dict]:
    placeholders = ",".join("?" for _ in statuses)
    order = " ORDER BY rank ASC" if order_by_rank else ""
    rows = conn.execute(
        f"SELECT * FROM revealed_candidates WHERE competition_id = ? AND status IN ({placeholders}){order}",
        (competition_id, *statuses),
    ).fetchall()
    return [dict(r) for r in rows]


@_locked
def count_candidates_by_status(conn: sqlite3.Connection, competition_id: str, statuses: tuple) -> int:
    placeholders = ",".join("?" for _ in statuses)
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM revealed_candidates WHERE competition_id = ? AND status IN ({placeholders})",
        (competition_id, *statuses),
    ).fetchone()
    return row["n"]


@_locked
def all_candidates_for_competition(conn: sqlite3.Connection, competition_id: str) -> List[dict]:
    rows = conn.execute(
        "SELECT * FROM revealed_candidates WHERE competition_id = ? ORDER BY rank ASC",
        (competition_id,),
    ).fetchall()
    return [dict(r) for r in rows]


@_locked
def insert_benchmark_result(
    conn: sqlite3.Connection,
    competition_id: str,
    hotkey: str,
    benchmark_name: str,
    repository: str,
    revision: str,
    coordinator_run_id: str,
    status: str = "submitted",
) -> None:
    now = time.time()
    with conn:
        conn.execute(
            """INSERT INTO benchmark_results
               (competition_id, hotkey, benchmark_name, repository, revision,
                coordinator_run_id, status, submitted_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(competition_id, hotkey, benchmark_name) DO UPDATE SET
                 repository = excluded.repository,
                 revision = excluded.revision,
                 coordinator_run_id = excluded.coordinator_run_id,
                 status = excluded.status,
                 updated_at = excluded.updated_at""",
            (competition_id, hotkey, benchmark_name, repository, revision, coordinator_run_id, status, now, now),
        )


@_locked
def update_benchmark_result(
    conn: sqlite3.Connection,
    competition_id: str,
    hotkey: str,
    benchmark_name: str,
    status: str,
    score: Optional[float] = None,
    phase: Optional[str] = None,
    percent_complete: Optional[float] = None,
    last_message: Optional[str] = None,
) -> None:
    with conn:
        conn.execute(
            """UPDATE benchmark_results
               SET status = ?, score = COALESCE(?, score), phase = ?, percent_complete = ?,
                   last_message = ?, updated_at = ?
               WHERE competition_id = ? AND hotkey = ? AND benchmark_name = ?""",
            (status, score, phase, percent_complete, last_message, time.time(),
             competition_id, hotkey, benchmark_name),
        )


@_locked
def open_benchmark_results(conn: sqlite3.Connection, competition_id: str) -> List[dict]:
    rows = conn.execute(
        """SELECT * FROM benchmark_results
           WHERE competition_id = ? AND status = 'submitted'""",
        (competition_id,),
    ).fetchall()
    return [dict(r) for r in rows]


@_locked
def benchmark_results_for_hotkey(conn: sqlite3.Connection, competition_id: str, hotkey: str) -> List[dict]:
    rows = conn.execute(
        "SELECT * FROM benchmark_results WHERE competition_id = ? AND hotkey = ?",
        (competition_id, hotkey),
    ).fetchall()
    return [dict(r) for r in rows]


@_locked
def record_scoring_result(
    conn: sqlite3.Connection,
    competition_id: str,
    hotkey: str,
    final_score: float,
    max_memory_kb: int,
) -> None:
    now = time.time()
    conn.execute(
        """INSERT INTO scoring_results (competition_id, hotkey, final_score, max_memory_kb, finalized_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(competition_id, hotkey) DO UPDATE SET
             final_score = excluded.final_score, max_memory_kb = excluded.max_memory_kb,
             finalized_at = excluded.finalized_at""",
        (competition_id, hotkey, final_score, max_memory_kb, now),
    )
    conn.commit()


@_locked
def finalize_candidate(
    conn: sqlite3.Connection,
    competition_id: str,
    hotkey: str,
    final_score: float,
    max_memory_kb: int,
) -> None:
    now = time.time()
    with conn:
        conn.execute(
            """INSERT INTO scoring_results (competition_id, hotkey, final_score, max_memory_kb, finalized_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(competition_id, hotkey) DO UPDATE SET
                 final_score = excluded.final_score, max_memory_kb = excluded.max_memory_kb,
                 finalized_at = excluded.finalized_at""",
            (competition_id, hotkey, final_score, max_memory_kb, now),
        )
        conn.execute(
            """UPDATE revealed_candidates SET status = 'done', failure_reason = NULL, updated_at = ?
               WHERE competition_id = ? AND hotkey = ?""",
            (now, competition_id, hotkey),
        )


@_locked
def scoring_results_for_competition(conn: sqlite3.Connection, competition_id: str) -> List[dict]:
    rows = conn.execute(
        "SELECT * FROM scoring_results WHERE competition_id = ? ORDER BY final_score DESC",
        (competition_id,),
    ).fetchall()
    return [dict(r) for r in rows]


@_locked
def record_weights(conn: sqlite3.Connection, competition_id: str, hotkey_weights: Dict[str, float]) -> None:
    conn.execute(
        "INSERT INTO weights_history (competition_id, set_at, weights_json) VALUES (?, ?, ?)",
        (competition_id, time.time(), json.dumps(hotkey_weights)),
    )
    conn.commit()


@_locked
def weights_history_for_competition(conn: sqlite3.Connection, competition_id: str) -> List[dict]:
    rows = conn.execute(
        "SELECT * FROM weights_history WHERE competition_id = ? ORDER BY id ASC",
        (competition_id,),
    ).fetchall()
    return [{"set_at": w["set_at"], "weights": json.loads(w["weights_json"])} for w in rows]


@_locked
def latest_weights_for_competition(conn: sqlite3.Connection, competition_id: str) -> Optional[Dict[str, float]]:
    """Most recently recorded per-hotkey weights for a competition, or None if never scored."""
    row = conn.execute(
        "SELECT weights_json FROM weights_history WHERE competition_id = ? ORDER BY id DESC LIMIT 1",
        (competition_id,),
    ).fetchone()
    return json.loads(row["weights_json"]) if row else None


@_locked
def full_state_for_competition(conn: sqlite3.Connection, competition_id: str) -> dict:
    candidates = all_candidates_for_competition(conn, competition_id)
    benchmark_rows = conn.execute(
        "SELECT * FROM benchmark_results WHERE competition_id = ?", (competition_id,)
    ).fetchall()
    results = scoring_results_for_competition(conn, competition_id)

    return {
        "competition_id": competition_id,
        "stage": get_stage(conn, competition_id),
        "candidates": candidates,
        "benchmark_results": [dict(r) for r in benchmark_rows],
        "results": results,
        "weights_history": weights_history_for_competition(conn, competition_id),
        "scored_status": scored_status(conn, competition_id),
    }


@_locked
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


@_locked
def get_competition(conn: sqlite3.Connection, competition_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT spec_json FROM competitions WHERE id = ?", (competition_id,)
    ).fetchone()
    return json.loads(row["spec_json"]) if row else None


@_locked
def list_competitions(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute("SELECT spec_json FROM competitions ORDER BY id ASC").fetchall()
    return [json.loads(r["spec_json"]) for r in rows]


@_locked
def competition_has_state(conn: sqlite3.Connection, competition_id: str) -> bool:
    """True if any reveal, benchmark, result, weight or stage row exists."""
    for table in ("revealed_candidates", "benchmark_results", "scoring_results",
                  "weights_history", "scored_competitions"):
        column = "competition_id"
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE {column} = ? LIMIT 1", (competition_id,)
        ).fetchone()
        if row is not None:
            return True
    return False


@_locked
def delete_competition(conn: sqlite3.Connection, competition_id: str) -> bool:
    """Remove a competition spec row. Returns False if it does not exist."""
    cur = conn.execute("DELETE FROM competitions WHERE id = ?", (competition_id,))
    conn.commit()
    return cur.rowcount > 0


@_locked
def ban(conn: sqlite3.Connection, hotkey: str, reason: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO banned_hotkeys (hotkey, reason, banned_at) VALUES (?, ?, ?)",
        (hotkey, reason, time.time()),
    )
    conn.commit()


@_locked
def is_banned(conn: sqlite3.Connection, hotkey: str) -> bool:
    row = conn.execute("SELECT 1 FROM banned_hotkeys WHERE hotkey = ?", (hotkey,)).fetchone()
    return row is not None
