import sqlite3

from common.models.competition import BenchmarkTask, CompetitionSpec
from common.models.submission import build_reveal_payload, Claim
from validator import store
from validator.chain_scanner import scan_reveals


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
    )


def make_payload(competition_id="comp1") -> str:
    return build_reveal_payload(
        competition_id=competition_id,
        repository="user/repo",
        file="model.gguf",
        file_sha256="a" * 64,
        max_memory=1000,
        claims=[Claim(b="mmlu", s=0.7)],
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


def test_scan_reveals_accepts_exact_block_match(monkeypatch):
    spec = make_spec(commit_end_block=100)
    reveals = {"hk1": [(make_payload(), 100)]}
    monkeypatch.setattr("validator.chain_scanner.read_revealed_commitments", lambda subtensor, netuid: reveals)

    result = scan_reveals(FakeSubtensor(reveals), spec, make_db())
    assert set(result) == {"hk1"}


def test_scan_reveals_rejects_block_before_window(monkeypatch):
    spec = make_spec(commit_end_block=100)
    reveals = {"hk1": [(make_payload(), 99)]}
    monkeypatch.setattr("validator.chain_scanner.read_revealed_commitments", lambda subtensor, netuid: reveals)

    result = scan_reveals(FakeSubtensor(reveals), spec, make_db())
    assert result == {}


def test_scan_reveals_accepts_within_grace_window(monkeypatch):
    spec = make_spec(commit_end_block=100)
    reveals = {"hk1": [(make_payload(), 103)]}
    monkeypatch.setattr("validator.chain_scanner.read_revealed_commitments", lambda subtensor, netuid: reveals)

    result = scan_reveals(FakeSubtensor(reveals), spec, make_db())
    assert set(result) == {"hk1"}


def test_scan_reveals_rejects_block_after_window(monkeypatch):
    spec = make_spec(commit_end_block=100)
    reveals = {"hk1": [(make_payload(), 105)]}
    monkeypatch.setattr("validator.chain_scanner.read_revealed_commitments", lambda subtensor, netuid: reveals)

    result = scan_reveals(FakeSubtensor(reveals), spec, make_db())
    assert result == {}


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
