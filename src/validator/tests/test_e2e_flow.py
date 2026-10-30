"""
End-to-end competition flow with only the benchmark coordinator faked.

Covers reveal, run-id verification, ranking, scoring and final weights in one
pass. The validator no longer downloads or runs anything itself — every number
it scores on comes from a coordinator run — so this needs no Docker, no network
and no GPU, and runs in the default suite.

Run:
    uv run python -m pytest src/validator/tests/test_e2e_flow.py -v
"""
import tempfile
from pathlib import Path

import pytest

from common import settings as common_settings
from common.models.competition import BenchmarkTask, CompetitionSpec
from common.models.submission import BenchmarkRun, MinerSubmission
from competition.benchmark_client import RunStatus, RunStatusCode
from validator import store
from validator.validator import Validator


# A small public GGUF pinned to an immutable revision, with the real LFS hash
# HuggingFace serves. Nothing is downloaded — these are the values the
# coordinator would report for this artifact.
REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
REVISION = "9217f5db79a29953eb74d5343926648285ec7e67"
FILE = "qwen2.5-0.5b-instruct-q2_k.gguf"
SHA256 = "9ee36184e616dfc76df4f5dd66f908dbde6979524ae36e6cefb67f532f798cb8"

# What the coordinator measured loading this model, and the competition's cap.
MEASURED_MEMORY_BYTES = 415_529_697
MEASURED_MEMORY_KB = MEASURED_MEMORY_BYTES // 1024   # 405,790 KB
MEMORY_CAP_KB = 1_000_000


# ---------------------------------------------------------------------------
# Fakes for everything outside the validator
# ---------------------------------------------------------------------------

class FakeWallet:
    def __init__(self, ss58="5ValidatorHotkey"):
        class _Hotkey:
            ss58_address = ss58
        self.hotkey = _Hotkey()


class FakeNeuron:
    def __init__(self, hotkey):
        self.hotkey = hotkey


class FakeMetagraph:
    def __init__(self, hotkeys):
        self.hotkeys = list(hotkeys)
        self.neurons = [FakeNeuron(hk) for hk in hotkeys]
        self.validators = []


class FakeSubnets:
    def __init__(self, metagraph):
        self._m = metagraph

    def metagraph(self, netuid):
        return self._m


class FakeSubtensor:
    def __init__(self, metagraph):
        self.subnets = FakeSubnets(metagraph)
        self.weights = type("W", (), {"weights": lambda self, netuid: {}})()

    def block(self):
        return 15


class ScriptedCoordinator:
    """Mock coordinator serving canned run statuses keyed by run id."""

    def __init__(self, statuses):
        self._statuses = statuses
        self.polled = []

    def list_benchmarks(self):
        return {"mmlu", "gsm8k"}

    def poll(self, run_id):
        self.polled.append(run_id)
        if run_id not in self._statuses:
            raise RuntimeError(f"unknown run id {run_id}")
        return self._statuses[run_id]


def run_status(run_id, score, *, repo=REPO, revision=REVISION, file=FILE, sha256=SHA256,
               benchmark="mmlu", memory_bytes=MEASURED_MEMORY_BYTES, **overrides):
    fields = dict(
        run_id=run_id,
        status=RunStatusCode.COMPLETED,
        scores={benchmark: score},
        repo=repo,
        revision=revision,
        model_files=[file],
        file_hashes={file: ("sha256", sha256)},
        benchmarks=[benchmark],
        item_status={benchmark: "completed"},
        measured_memory_bytes=memory_bytes,
    )
    fields.update(overrides)
    return RunStatus(**fields)


def make_submission(run_id, *, file_sha256=SHA256, file=FILE):
    return MinerSubmission(
        competition_id="e2e",
        runs=[BenchmarkRun(b="mmlu", r=run_id)],
        repository=REPO,
        file=file,
        file_sha256=file_sha256,
        huggingface_revision=REVISION,
    )


def make_spec(**overrides):
    fields = dict(
        id="e2e", name="E2E competition",
        start_block=0, commit_end_block=10, scoring_end_block=1000,
        reveal_grace_blocks=0,
        top_n=1, emission_distribution=[1.0],
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.3)],
        competition_type="gpu_memory_ceiling", max_gpu_memory_kb=MEMORY_CAP_KB,
    )
    fields.update(overrides)
    return CompetitionSpec(**fields)


@pytest.fixture
def validator(monkeypatch):
    monkeypatch.setattr(common_settings, "BITTENSOR", False)
    monkeypatch.setattr(common_settings, "VALIDATOR_MODE", "leader", raising=False)
    db_path = Path(tempfile.mkstemp(suffix=".db")[1])
    monkeypatch.setattr(store, "validator_db_path", lambda: db_path)

    metagraph = FakeMetagraph(["hk_alpha", "hk_beta"])
    return Validator(
        wallet=FakeWallet(),
        subtensor=FakeSubtensor(metagraph),
        metagraph=metagraph,
    )


def scan(monkeypatch, reveals):
    monkeypatch.setattr(
        "validator.chain_scanner.scan_reveals", lambda subtensor, spec: reveals,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_flow_verified_run_scores_and_wins(validator, monkeypatch):
    """The happy path: a miner whose run verifies is ranked, scored on the
    coordinator's own numbers, and paid — all within one stage."""
    v = validator
    spec = make_spec()
    coordinator = ScriptedCoordinator({"r1000": run_status("r1000", 0.81)})
    monkeypatch.setattr(v, "_get_coordinator", lambda: coordinator)
    scan(monkeypatch, {"hk_alpha": (make_submission("r1000"), 5)})

    await v.run_stage_1(spec)

    row = store.benchmark_results_for_hotkey(v._db, "e2e", "hk_alpha")[0]
    assert row["status"] == "completed"
    assert row["score"] == 0.81

    candidate = store.get_candidate(v._db, "e2e", "hk_alpha")
    assert candidate["status"] == "done", candidate["failure_reason"]
    # The memory figure is the coordinator's measurement, not a miner claim.
    assert candidate["measured_memory_kb"] == MEASURED_MEMORY_KB

    results = store.scoring_results_for_competition(v._db, "e2e")
    assert [r["hotkey"] for r in results] == ["hk_alpha"]
    assert store.latest_weights_for_competition(v._db, "e2e") == {"hk_alpha": 1.0}


@pytest.mark.asyncio
async def test_full_flow_unverified_run_loses_to_verified_one(validator, monkeypatch):
    """The central guarantee: a miner whose run benchmarked a different repo
    scores 0.0 and cannot outrank an honest miner, however high the run's
    reported score."""
    v = validator
    spec = make_spec()
    # The two miners must commit different file hashes, or cross-hotkey
    # sha256 dedup eliminates one before verification ever runs.
    cheat_sha = "c" * 64
    coordinator = ScriptedCoordinator({
        # r2001 (hk_beta) reports a near-perfect score but benchmarked
        # another repo; r2002 (hk_alpha) is the honest, lower-scoring run.
        "r2001": run_status("r2001", 0.99, repo="someone-else/strong-model", sha256=cheat_sha),
        "r2002": run_status("r2002", 0.55),
    })
    monkeypatch.setattr(v, "_get_coordinator", lambda: coordinator)
    scan(monkeypatch, {
        "hk_beta": (make_submission("r2001", file_sha256=cheat_sha), 5),
        "hk_alpha": (make_submission("r2002"), 6),
    })

    await v.run_stage_1(spec)

    # Honest miner ranks first despite reporting the lower score.
    assert store.get_candidate(v._db, "e2e", "hk_alpha")["rank"] == 0
    assert store.get_candidate(v._db, "e2e", "hk_beta")["rank"] == 1

    cheat_row = store.benchmark_results_for_hotkey(v._db, "e2e", "hk_beta")[0]
    assert cheat_row["status"] == "failed"
    assert cheat_row["score"] == 0.0
    assert "repo mismatch" in cheat_row["last_message"]
    assert cheat_row["repository"] == "someone-else/strong-model"

    assert store.get_candidate(v._db, "e2e", "hk_alpha")["status"] == "done"
    assert store.latest_weights_for_competition(v._db, "e2e") == {"hk_alpha": 1.0}


@pytest.mark.asyncio
async def test_full_flow_sha256_mismatch_scores_zero_without_banning(validator, monkeypatch):
    """A commit whose sha256 contradicts the hash the coordinator reports for
    the file fails that run. Nothing bans the hotkey: the validator no longer
    hashes the file itself, so a mismatch is a rejected run, not proven
    misconduct."""
    v = validator
    spec = make_spec()
    coordinator = ScriptedCoordinator({"r2003": run_status("r2003", 0.9, sha256="d" * 64)})
    monkeypatch.setattr(v, "_get_coordinator", lambda: coordinator)
    scan(monkeypatch, {"hk_alpha": (make_submission("r2003"), 5)})

    await v.run_stage_1(spec)

    row = store.benchmark_results_for_hotkey(v._db, "e2e", "hk_alpha")[0]
    assert row["status"] == "failed"
    assert "file hash mismatch" in row["last_message"]

    candidate = store.get_candidate(v._db, "e2e", "hk_alpha")
    assert candidate["status"] == "failed"
    assert "failed floors" in candidate["failure_reason"]
    assert store.scored_status(v._db, "e2e") == "failed_no_participants"


@pytest.mark.asyncio
async def test_full_flow_memory_cap_rejects_oversized_model(validator, monkeypatch):
    """A measured figure above the competition's cap fails the candidate."""
    v = validator
    spec = make_spec(max_gpu_memory_kb=MEASURED_MEMORY_KB // 2)
    coordinator = ScriptedCoordinator({"r1000": run_status("r1000", 0.9)})
    monkeypatch.setattr(v, "_get_coordinator", lambda: coordinator)
    scan(monkeypatch, {"hk_alpha": (make_submission("r1000"), 5)})

    await v.run_stage_1(spec)

    candidate = store.get_candidate(v._db, "e2e", "hk_alpha")
    assert candidate["status"] == "failed"
    assert "exceeded memory cap" in candidate["failure_reason"], candidate["failure_reason"]


@pytest.mark.asyncio
async def test_full_flow_run_without_a_memory_figure_is_not_scored(validator, monkeypatch):
    """A coordinator run predating the `vram` block reports no memory. Scoring
    it as 0 KB would pass any cap, so the candidate is failed instead."""
    v = validator
    coordinator = ScriptedCoordinator({"r1000": run_status("r1000", 0.9, memory_bytes=None)})
    monkeypatch.setattr(v, "_get_coordinator", lambda: coordinator)
    scan(monkeypatch, {"hk_alpha": (make_submission("r1000"), 5)})

    await v.run_stage_1(make_spec())

    candidate = store.get_candidate(v._db, "e2e", "hk_alpha")
    assert candidate["status"] == "failed"
    assert "GPU memory" in candidate["failure_reason"]
    assert store.scoring_results_for_competition(v._db, "e2e") == []
