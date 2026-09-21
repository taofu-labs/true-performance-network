"""
End-to-end competition flow: mock coordinator, REAL precheck Docker container.

Covers the full rework in one pass — reveal, run-id verification, ranking,
precheck against a real llama-cli, and final weights — with only the benchmark
coordinator faked. The precheck container downloads a real GGUF from
HuggingFace and measures its RAM for real, which is the half that unit tests
deliberately stub out.

Run:
    uv run python -m pytest src/validator/tests/test_e2e_flow.py -m integration -v -s

Requires Docker, network access, and the precheck image:
    docker build --platform linux/arm64 -t tpn-precheck:test -f shared/validation/precheck.Dockerfile shared/validation
Point at another tag with PRECHECK_IMAGE.
"""
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from common import settings as common_settings
from common.models.competition import BenchmarkTask, CompetitionSpec
from common.models.submission import BenchmarkRun, MinerSubmission
from competition.benchmark_client import RunStatus, RunStatusCode
from validator import store
from validator.validator import Validator

pytestmark = pytest.mark.integration


# A small public GGUF pinned to an immutable revision. The sha256 is the real
# LFS hash HuggingFace serves, so the hash-match path is exercised for real
# rather than against a fixture value.
REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
REVISION = "9217f5db79a29953eb74d5343926648285ec7e67"
FILE = "qwen2.5-0.5b-instruct-q2_k.gguf"
SHA256 = "9ee36184e616dfc76df4f5dd66f908dbde6979524ae36e6cefb67f532f798cb8"

# max_memory is a precise self-report, not a ceiling: precheck fails a
# candidate whose claim differs from the measured value by more than
# RAM_CHECK_LYING_TOLERANCE (1%). This is the real measured figure for this
# model at context_length=512, rounded down as the validator does.
MEASURED_MEMORY_KB = 415_529_697 // 1024   # 405,790 KB
MAX_MEMORY_KB = MEASURED_MEMORY_KB

# The competition's cap, comfortably above the real measurement.
MEMORY_CAP_KB = 1_000_000

IMAGE = os.environ.get("PRECHECK_IMAGE", "tpn-precheck:test")


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=15, check=True)
    except Exception:
        return False
    probe = subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True, timeout=15
    )
    return probe.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason=f"Docker unavailable or image {IMAGE!r} not built",
)


# ---------------------------------------------------------------------------
# Fakes for everything that is not the precheck container
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
               benchmark="mmlu", **overrides):
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
    )
    fields.update(overrides)
    return RunStatus(**fields)


def make_submission(run_id, *, max_memory=MAX_MEMORY_KB, file_sha256=SHA256, file=FILE):
    return MinerSubmission(
        competition_id="e2e",
        runs=[BenchmarkRun(b="mmlu", r=run_id)],
        repository=REPO,
        file=file,
        file_sha256=file_sha256,
        max_memory=max_memory,
        huggingface_revision=REVISION,
    )


def make_spec(**overrides):
    fields = dict(
        id="e2e", name="E2E competition",
        start_block=0, commit_end_block=10, scoring_end_block=1000,
        reveal_grace_blocks=0,
        top_n=1, emission_distribution=[1.0],
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.3, weight=1.0)],
        competition_type="ram_ceiling", max_memory_kb=MEMORY_CAP_KB,
        ram_check_context_length=512,   # keep the real llama-cli run quick
        model_repo=None,                # no base model -> no provenance download
    )
    fields.update(overrides)
    return CompetitionSpec(**fields)


@pytest.fixture
def validator(monkeypatch):
    monkeypatch.setattr(common_settings, "BITTENSOR", False)
    monkeypatch.setattr(common_settings, "VALIDATOR_MODE", "leader", raising=False)
    monkeypatch.setenv("PRECHECK_IMAGE", IMAGE)
    db_path = Path(tempfile.mkstemp(suffix=".db")[1])
    monkeypatch.setattr(store, "validator_db_path", lambda: db_path)

    metagraph = FakeMetagraph(["hk_alpha", "hk_beta"])
    v = Validator(
        wallet=FakeWallet(),
        subtensor=FakeSubtensor(metagraph),
        metagraph=metagraph,
    )
    yield v
    subprocess.run(["docker", "rm", "-f", f"tpn-precheck-{make_spec().id}"],
                   capture_output=True, timeout=30)


async def drive_stage_2(v, spec, max_ticks=40, tick_delay=2):
    """Pump stage 2 until the competition finalizes, as the leader loop would."""
    import asyncio

    for tick in range(max_ticks):
        if store.is_scored(v._db, spec.id):
            return tick
        await v.run_stage_2(spec, current_block=15)
        if store.is_scored(v._db, spec.id):
            return tick
        await asyncio.sleep(tick_delay)
    raise AssertionError(f"stage 2 did not finalize within {max_ticks} ticks")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_docker
@pytest.mark.asyncio
async def test_full_flow_verified_run_scores_and_wins(validator, monkeypatch):
    """The happy path end to end: a miner whose run verifies is ranked,
    prechecked against a real llama-cli, scored, and paid."""
    v = validator
    spec = make_spec()
    coordinator = ScriptedCoordinator({"r1000": run_status("r1000", 0.81)})
    monkeypatch.setattr(v, "_get_coordinator", lambda: coordinator)

    submission = make_submission("r1000")
    monkeypatch.setattr(
        "validator.chain_scanner.scan_reveals",
        lambda subtensor, spec, db: {"hk_alpha": (submission, 5)},
    )

    await v.run_stage_1(spec)

    assert store.get_stage(v._db, "e2e") == "stage2_scoring"
    candidate = store.get_candidate(v._db, "e2e", "hk_alpha")
    assert candidate["status"] == "standby"
    row = store.benchmark_results_for_hotkey(v._db, "e2e", "hk_alpha")[0]
    assert row["status"] == "completed"
    assert row["score"] == 0.81

    ticks = await drive_stage_2(v, spec)
    print(f"\n[e2e] finalized after {ticks} stage-2 tick(s)")

    candidate = store.get_candidate(v._db, "e2e", "hk_alpha")
    assert candidate["status"] == "done", candidate["failure_reason"]
    assert candidate["gguf_file"] == FILE

    # The RAM figure came from a real llama-cli load, not a stub.
    measured = candidate["measured_memory_kb"]
    assert 0 < measured < MEMORY_CAP_KB
    print(f"[e2e] measured RAM: {measured:,} KB")

    results = store.scoring_results_for_competition(v._db, "e2e")
    assert [r["hotkey"] for r in results] == ["hk_alpha"]
    assert store.latest_weights_for_competition(v._db, "e2e") == {"hk_alpha": 1.0}
    assert store.is_banned(v._db, "hk_alpha") is False


@requires_docker
@pytest.mark.asyncio
async def test_full_flow_unverified_run_loses_to_verified_one(validator, monkeypatch):
    """The rework's central guarantee, end to end: a miner whose run
    benchmarked a different repo scores 0.0 and cannot outrank an honest
    miner, however high the run's reported score."""
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

    monkeypatch.setattr(
        "validator.chain_scanner.scan_reveals",
        lambda subtensor, spec, db: {
            "hk_beta": (make_submission("r2001", file_sha256=cheat_sha), 5),
            "hk_alpha": (make_submission("r2002"), 6),
        },
    )

    await v.run_stage_1(spec)

    # Honest miner ranks first despite claiming the lower score.
    assert store.get_candidate(v._db, "e2e", "hk_alpha")["rank"] == 0
    assert store.get_candidate(v._db, "e2e", "hk_beta")["rank"] == 1

    cheat_row = store.benchmark_results_for_hotkey(v._db, "e2e", "hk_beta")[0]
    assert cheat_row["status"] == "failed"
    assert cheat_row["score"] == 0.0
    assert "repo mismatch" in cheat_row["last_message"]
    assert cheat_row["repository"] == "someone-else/strong-model"

    await drive_stage_2(v, spec)

    assert store.get_candidate(v._db, "e2e", "hk_alpha")["status"] == "done"
    assert store.latest_weights_for_competition(v._db, "e2e") == {"hk_alpha": 1.0}


@requires_docker
@pytest.mark.asyncio
async def test_full_flow_sha256_mismatch_bans_hotkey(validator, monkeypatch):
    """A miner committing a sha256 that does not match the real file is
    banned by the real precheck container, which hashes the actual download."""
    v = validator
    spec = make_spec()
    wrong_sha = "d" * 64
    coordinator = ScriptedCoordinator({
        # The run's own hash matches the commit, so verification passes and
        # only the real download can catch the lie.
        "r2003": run_status("r2003", 0.9, sha256=wrong_sha),
    })
    monkeypatch.setattr(v, "_get_coordinator", lambda: coordinator)
    monkeypatch.setattr(
        "validator.chain_scanner.scan_reveals",
        lambda subtensor, spec, db: {"hk_alpha": (make_submission("r2003", file_sha256=wrong_sha), 5)},
    )

    await v.run_stage_1(spec)
    assert store.benchmark_results_for_hotkey(v._db, "e2e", "hk_alpha")[0]["status"] == "completed"

    await drive_stage_2(v, spec)

    candidate = store.get_candidate(v._db, "e2e", "hk_alpha")
    assert candidate["status"] == "failed"
    assert "sha256 mismatch" in candidate["failure_reason"]
    assert store.is_banned(v._db, "hk_alpha") is True
    assert store.scored_status(v._db, "e2e") == "failed_no_participants"


@requires_docker
@pytest.mark.asyncio
async def test_full_flow_memory_cap_rejects_oversized_model(validator, monkeypatch):
    """A real RAM measurement above the competition's cap fails the
    candidate. The cap is set below what the model actually needs."""
    v = validator
    # Cap set below what the model really needs, but the miner reports its
    # memory honestly — so only the cap can reject it, not the lie check.
    spec = make_spec(max_memory_kb=MEASURED_MEMORY_KB // 2)
    coordinator = ScriptedCoordinator({"r1000": run_status("r1000", 0.9)})
    monkeypatch.setattr(v, "_get_coordinator", lambda: coordinator)
    monkeypatch.setattr(
        "validator.chain_scanner.scan_reveals",
        lambda subtensor, spec, db: {"hk_alpha": (make_submission("r1000"), 5)},
    )

    await v.run_stage_1(spec)
    await drive_stage_2(v, spec)

    candidate = store.get_candidate(v._db, "e2e", "hk_alpha")
    assert candidate["status"] == "failed"
    assert "exceeded memory cap" in candidate["failure_reason"], candidate["failure_reason"]
    # An oversized model is a losing entry, not misconduct.
    assert store.is_banned(v._db, "hk_alpha") is False
