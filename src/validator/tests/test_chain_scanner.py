import sqlite3

import pytest

from common.models.competition import BenchmarkTask, CompetitionSpec
from common.models.submission import BenchmarkRun, build_reveal_payload
from validator import store
from validator.chain_scanner import scan_reveals


GRACE = 15


def make_spec(commit_end_block=100) -> CompetitionSpec:
    return CompetitionSpec(
        id="comp1",
        name="comp1",
        start_block=0,
        commit_end_block=commit_end_block,
        scoring_end_block=200,
        emission_distribution=[1.0],
        top_n=1,
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.5, weight=1.0)],
        reveal_grace_blocks=GRACE,
    )


def make_payload(competition_id="comp1") -> str:
    return build_reveal_payload(
        competition_id=competition_id,
        repository="user/repo",
        file="model.gguf",
        file_sha256="a" * 64,
        max_memory=1000,
        runs=[BenchmarkRun(b="mmlu", r="r100")],
        huggingface_revision="a" * 40,
    )


class FakeSubtensor:
    def __init__(self, reveals):
        self._reveals = reveals


def make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(store._SCHEMA)
    return conn


@pytest.mark.parametrize("reveal_block,accepted", [
    (100,             True),   # exactly commit_end_block
    (100 - GRACE,     True),   # window is symmetric — grace_blocks early still counts
    (100 - GRACE - 1, False),  # one block before the window opens
    (100 + GRACE - 1, True),   # last accepted block
    (100 + GRACE,     False),  # window is exclusive at the top
])
def test_scan_reveals_block_window(monkeypatch, reveal_block, accepted):
    spec = make_spec(commit_end_block=100)
    reveals = {"hk1": [(make_payload(), reveal_block)]}
    monkeypatch.setattr("validator.chain_scanner.read_revealed_commitments", lambda subtensor, netuid: reveals)

    result = scan_reveals(FakeSubtensor(reveals), spec, make_db())
    assert set(result) == ({"hk1"} if accepted else set())


def test_scan_reveals_skips_banned_hotkey(monkeypatch):
    spec = make_spec(commit_end_block=100)
    reveals = {"hk1": [(make_payload(), 100)]}
    monkeypatch.setattr("validator.chain_scanner.read_revealed_commitments", lambda subtensor, netuid: reveals)

    conn = make_db()
    store.ban(conn, "hk1", "test ban")
    result = scan_reveals(FakeSubtensor(reveals), spec, conn)
    assert result == {}


def test_scan_reveals_skips_wrong_competition_id(monkeypatch):
    spec = make_spec(commit_end_block=100)
    reveals = {"hk1": [(make_payload(competition_id="other-comp"), 100)]}
    monkeypatch.setattr("validator.chain_scanner.read_revealed_commitments", lambda subtensor, netuid: reveals)

    result = scan_reveals(FakeSubtensor(reveals), spec, make_db())
    assert result == {}


def test_scan_reveals_skips_malformed_payload(monkeypatch):
    spec = make_spec(commit_end_block=100)
    reveals = {"hk1": [("not json", 100)]}
    monkeypatch.setattr("validator.chain_scanner.read_revealed_commitments", lambda subtensor, netuid: reveals)

    result = scan_reveals(FakeSubtensor(reveals), spec, make_db())
    assert result == {}


def test_scan_reveals_returns_reveal_block(monkeypatch):
    spec = make_spec(commit_end_block=100)
    reveals = {"hk1": [(make_payload(), 103)]}
    monkeypatch.setattr("validator.chain_scanner.read_revealed_commitments", lambda subtensor, netuid: reveals)

    result = scan_reveals(FakeSubtensor(reveals), spec, make_db())
    submission, reveal_block = result["hk1"]
    assert reveal_block == 103
    assert submission.competition_id == "comp1"
