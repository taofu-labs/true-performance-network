import asyncio
import tempfile
from pathlib import Path

import pytest

from common import settings as common_settings
from common.models.competition import BenchmarkTask, CompetitionSpec
from validator import settings as validator_settings
from validator import store
from validator.validator import Validator


class FakeWallet:
    def __init__(self, ss58="5FakeHotkey"):
        class _Hotkey:
            ss58_address = ss58
        self.hotkey = _Hotkey()


class FakeValidatorNeuron:
    def __init__(self, uid, alpha_stake):
        self.uid = uid

        class _Stake:
            alpha = alpha_stake
        self.total_stake = _Stake()


class FakeMetagraph:
    def __init__(self, hotkeys, uids, stake, weights, validator_permit):
        self.hotkeys = hotkeys
        self.validators = [
            FakeValidatorNeuron(uid, s)
            for uid, s, permit in zip(uids, stake, validator_permit)
            if permit
        ]
        self._weight_rows = {uid: dict(zip(uids, row)) for uid, row in zip(uids, weights)}


class FakeWeightsNamespace:
    def __init__(self, rows):
        self._rows = rows

    def weights(self, netuid):
        return self._rows


class FakeSubnetsNamespace:
    def __init__(self, metagraph):
        self._metagraph = metagraph

    def metagraph(self, netuid):
        return self._metagraph


class FakeSubtensor:
    def __init__(self, metagraph=None):
        self._metagraph = metagraph
        self.subnets = FakeSubnetsNamespace(metagraph)
        self.weights = FakeWeightsNamespace(metagraph._weight_rows if metagraph else {})

    def block(self):
        return 123


def make_validator(monkeypatch, metagraph=None, mode="leader"):
    monkeypatch.setattr(common_settings, "BITTENSOR", False)
    monkeypatch.setattr(common_settings, "VALIDATOR_MODE", mode, raising=False)
    # Each test gets its own DB file — store.init_db() caches connections by
    # path at module scope, so without this every test in this file would
    # share (and pollute) the same on-disk validator.db.
    db_path = Path(tempfile.mkstemp(suffix=".db")[1])
    monkeypatch.setattr(store, "validator_db_path", lambda: db_path)
    return Validator(
        wallet=FakeWallet(),
        subtensor=FakeSubtensor(metagraph=metagraph),
        metagraph=metagraph,
    )



def test_copy_weights_from_chain_stake_weighted_average(monkeypatch):
    metagraph = FakeMetagraph(
        hotkeys=["hk0", "hk1"],
        uids=[0, 1],
        stake=[1.0, 3.0],
        weights=[[0.5, 0.5], [0.2, 0.8]],
        validator_permit=[True, True],
    )
    v = make_validator(monkeypatch, metagraph=metagraph)
    result = v.copy_weights_from_chain()
    # validator 0 (stake 1, weight 0.25) row [0.5, 0.5]; validator 1 (stake 3, weight 0.75) row [0.2, 0.8]
    assert result[0] == pytest.approx(0.25 * 0.5 + 0.75 * 0.2)
    assert result[1] == pytest.approx(0.25 * 0.5 + 0.75 * 0.8)



@pytest.mark.asyncio
async def test_set_weights_skips_when_bittensor_disabled(monkeypatch):
    v = make_validator(monkeypatch)
    monkeypatch.setattr(common_settings, "BITTENSOR", False)
    await v.set_weights(weights={0: 1.0})  # must not raise, no wallet/subtensor calls needed



@pytest.mark.asyncio
async def test_compute_and_set_aggregate_weights_no_distributing_competitions(monkeypatch):
    v = make_validator(monkeypatch)
    monkeypatch.setattr(v, "_get_active_competitions", lambda current_block: [])
    distributed = await v._compute_and_set_aggregate_weights(current_block=123)
    assert distributed is False


class FakeSpec:
    def __init__(self, id, emission_weight, distributing=True):
        self.id = id
        self.emission_weight = emission_weight
        self._distributing = distributing

    def is_distributing(self, current_block):
        return self._distributing


@pytest.mark.asyncio
async def test_compute_and_set_aggregate_weights_burns_shortfall_to_uid0(monkeypatch):
    metagraph = FakeMetagraph(
        hotkeys=["hk0", "hk1"], uids=[0, 1], stake=[1.0], weights=[[1.0]], validator_permit=[True],
    )
    v = make_validator(monkeypatch, metagraph=metagraph)
    spec = FakeSpec(id="comp", emission_weight=0.5)
    monkeypatch.setattr(v, "_get_active_competitions", lambda current_block: [spec])
    monkeypatch.setattr(
        "validator.store.latest_weights_for_competition",
        lambda db, comp_id: {"hk1": 0.6},
    )
    captured = {}

    async def fake_set_weights(weights):
        captured.update(weights)
    monkeypatch.setattr(v, "set_weights", fake_set_weights)

    distributed = await v._compute_and_set_aggregate_weights(current_block=123)
    assert distributed is True
    # hk1 -> uid 1 gets emission_weight * share = 0.5 * 0.6 = 0.3; shortfall 0.7 burned to uid 0
    assert captured[1] == pytest.approx(0.3)
    assert captured[0] == pytest.approx(0.7)
    assert sum(captured.values()) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_compute_and_set_aggregate_weights_no_burn_when_full(monkeypatch):
    metagraph = FakeMetagraph(
        hotkeys=["hk0", "hk1"], uids=[0, 1], stake=[1.0], weights=[[1.0]], validator_permit=[True],
    )
    v = make_validator(monkeypatch, metagraph=metagraph)
    spec = FakeSpec(id="comp", emission_weight=1.0)
    monkeypatch.setattr(v, "_get_active_competitions", lambda current_block: [spec])
    monkeypatch.setattr(
        "validator.store.latest_weights_for_competition",
        lambda db, comp_id: {"hk1": 1.0},
    )
    captured = {}

    async def fake_set_weights(weights):
        captured.update(weights)
    monkeypatch.setattr(v, "set_weights", fake_set_weights)

    distributed = await v._compute_and_set_aggregate_weights(current_block=123)
    assert distributed is True
    assert captured == {1: pytest.approx(1.0)}
    assert 0 not in captured


def make_spec(**overrides):
    fields = {
        "id": "comp1", "name": "comp1", "start_block": 0, "commit_end_block": 10,
        "scoring_end_block": 20, "emission_distribution": [1.0], "top_n": 1,
        "benchmarks": [BenchmarkTask(name="mmlu", min_score=0.5)],
    }
    fields.update(overrides)
    return CompetitionSpec.model_validate(fields)


class FakeNeuron:
    def __init__(self, hotkey):
        self.hotkey = hotkey


def make_submission(**overrides):
    from common.models.submission import BenchmarkRun, MinerSubmission
    fields = dict(
        competition_id="comp1", runs=[BenchmarkRun(b="mmlu", r="r100")], repository="user/repo",
        file="model.gguf", file_sha256="a" * 64, huggingface_revision="a" * 40,
    )
    fields.update(overrides)
    return MinerSubmission(**fields)


def make_run_status(**overrides):
    """A coordinator run that verifies cleanly against make_submission()."""
    from competition.benchmark_client import RunStatus, RunStatusCode
    fields = dict(
        run_id="r100", status=RunStatusCode.COMPLETED, scores={"mmlu": 0.8},
        repo="user/repo", revision="a" * 40, model_files=["model.gguf"],
        file_hashes={"model.gguf": ("sha256", "a" * 64)},
        benchmarks=["mmlu"], item_status={"mmlu": "completed"},
        measured_memory_bytes=900 * 1024,
    )
    fields.update(overrides)
    return RunStatus(**fields)


def stub_coordinator(monkeypatch, statuses=None, benchmarks={"mmlu"}):
    """Patch make_coordinator with one that serves canned run statuses."""
    statuses = statuses if statuses is not None else {"r100": make_run_status()}

    class Stub:
        def list_benchmarks(self):
            return benchmarks

        def poll(self, run_id):
            if run_id not in statuses:
                raise RuntimeError(f"unknown run {run_id}")
            return statuses[run_id]

    monkeypatch.setattr("competition.benchmark_client.make_coordinator", lambda: Stub())


# ---------------------------------------------------------------------------
# run_stage_1
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_stage_1_terminal_when_no_reveals(monkeypatch):
    """No reveals is a terminal, non-retryable outcome — reveals are
    chain-provably immutable once the commit window closes."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    monkeypatch.setattr("validator.chain_scanner.scan_reveals", lambda subtensor, spec: {})

    await v.run_stage_1(make_spec())
    assert store.scored_status(v._db, "comp1") == "failed_no_reveals"
    assert store.is_scored(v._db, "comp1") is True


@pytest.mark.asyncio
async def test_run_stage_1_retries_on_infra_failure(monkeypatch):
    """A chain-read exception (not an empty result) must retry, capped at
    STAGE1_MAX_ATTEMPTS, distinct from the terminal no-reveals path."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)

    def boom(subtensor, spec):
        raise RuntimeError("chain RPC failed")
    monkeypatch.setattr("validator.chain_scanner.scan_reveals", boom)

    await v.run_stage_1(make_spec())
    assert store.get_stage(v._db, "comp1") == "stage1_ranking"  # not terminal yet
    assert store.is_scored(v._db, "comp1") is False

    await v.run_stage_1(make_spec())
    await v.run_stage_1(make_spec())
    assert store.scored_status(v._db, "comp1") == "failed_stage1_infra"


@pytest.mark.asyncio
async def test_run_stage_1_persists_ranked_candidates(monkeypatch):
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    metagraph.neurons = [FakeNeuron("hk1")]
    v = make_validator(monkeypatch, metagraph=metagraph)

    submission = make_submission()
    monkeypatch.setattr("validator.chain_scanner.scan_reveals", lambda subtensor, spec: {"hk1": (submission, 5)})
    stub_coordinator(monkeypatch)

    await v.run_stage_1(make_spec())

    # Ranked, scored and finalized in the one pass — there is no second stage.
    candidate = store.get_candidate(v._db, "comp1", "hk1")
    assert candidate["status"] == "done"
    assert candidate["rank"] == 0
    assert store.get_stage(v._db, "comp1") == "finalized"


@pytest.mark.asyncio
async def test_run_stage_1_ranks_by_verified_score_not_by_claim(monkeypatch):
    """The point of the rework. Both miners commit the same memory, so rank is
    decided by the benchmark composite — and that now comes from the
    coordinator, so a miner cannot buy rank by claiming a high number."""
    metagraph = FakeMetagraph(hotkeys=["hk_low", "hk_high"], uids=[0, 1], stake=[1.0, 1.0],
                              weights=[[1.0, 0.0], [0.0, 1.0]], validator_permit=[True, True])
    metagraph.neurons = [FakeNeuron("hk_low"), FakeNeuron("hk_high")]
    v = make_validator(monkeypatch, metagraph=metagraph)

    from common.models.submission import BenchmarkRun
    low = make_submission(runs=[BenchmarkRun(b="mmlu", r="r1")], file_sha256="1" * 64)
    high = make_submission(runs=[BenchmarkRun(b="mmlu", r="r2")], file_sha256="2" * 64)
    monkeypatch.setattr(
        "validator.chain_scanner.scan_reveals",
        lambda subtensor, spec: {"hk_low": (low, 5), "hk_high": (high, 5)},
    )
    stub_coordinator(monkeypatch, statuses={
        "r1": make_run_status(run_id="r1", scores={"mmlu": 0.51}, file_hashes={"model.gguf": ("sha256", "1" * 64)}),
        "r2": make_run_status(run_id="r2", scores={"mmlu": 0.95}, file_hashes={"model.gguf": ("sha256", "2" * 64)}),
    })

    await v.run_stage_1(make_spec(competition_type="gpu_memory_ceiling", max_gpu_memory_kb=10_000))

    assert store.get_candidate(v._db, "comp1", "hk_high")["rank"] == 0
    assert store.get_candidate(v._db, "comp1", "hk_low")["rank"] == 1



@pytest.mark.asyncio
async def test_run_stage_1_keeps_candidate_whose_runs_all_failed_verification(monkeypatch):
    """A candidate scoring 0.0 everywhere is kept with a record of why, not
    dropped — the floors reject it later, exactly like a weak model."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    metagraph.neurons = [FakeNeuron("hk1")]
    v = make_validator(monkeypatch, metagraph=metagraph)

    monkeypatch.setattr("validator.chain_scanner.scan_reveals", lambda subtensor, spec: {"hk1": (make_submission(), 5)})
    stub_coordinator(monkeypatch, statuses={"r100": make_run_status(repo="someone/else")})

    await v.run_stage_1(make_spec())

    # Kept with a record of why, then failed by the floors rather than dropped.
    candidate = store.get_candidate(v._db, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "failed floors" in candidate["failure_reason"]
    row = store.benchmark_results_for_hotkey(v._db, "comp1", "hk1")[0]
    assert row["status"] == "failed"
    assert "repo mismatch" in row["last_message"]


@pytest.mark.asyncio
async def test_run_stage_1_retries_when_coordinator_unreachable_during_verification(monkeypatch):
    """A coordinator outage must not zero-score the whole field — it is infra
    failure, retried like a chain RPC error."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    metagraph.neurons = [FakeNeuron("hk1")]
    v = make_validator(monkeypatch, metagraph=metagraph)

    monkeypatch.setattr("validator.chain_scanner.scan_reveals", lambda subtensor, spec: {"hk1": (make_submission(), 5)})
    stub_coordinator(monkeypatch)
    monkeypatch.setattr(
        "validator.scorer.verify_all_candidates",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("coordinator unreachable")),
    )

    await v.run_stage_1(make_spec())

    assert store.get_stage(v._db, "comp1") == "stage1_ranking"
    assert store.is_scored(v._db, "comp1") is False
    assert store.get_candidate(v._db, "comp1", "hk1") is None


@pytest.mark.asyncio
async def test_run_stage_1_dedup_tie_break_demotes_displaced_winner(monkeypatch):
    """Two reveals at the identical reveal_block with the same file_sha256:
    dedup_winner's tiebreak (lower hotkey wins) must not just skip inserting
    the later-processed loser — it must also demote whichever hotkey was
    sitting in seen_hashes when it gets displaced. hk_z is processed first
    (dict insertion order) and occupies seen_hashes; hk_a is processed second
    and displaces it on the tiebreak (hk_a < hk_z). hk_z must end up as the
    loser, not silently pass through as a second standby candidate."""
    metagraph = FakeMetagraph(hotkeys=["hk_z", "hk_a"], uids=[0, 1], stake=[1.0, 1.0],
                               weights=[[1.0, 0.0], [0.0, 1.0]], validator_permit=[True, True])
    metagraph.neurons = [FakeNeuron("hk_z"), FakeNeuron("hk_a")]
    v = make_validator(monkeypatch, metagraph=metagraph)

    same_sha = "b" * 64
    submission_z = make_submission(file_sha256=same_sha)
    submission_a = make_submission(file_sha256=same_sha)
    monkeypatch.setattr(
        "validator.chain_scanner.scan_reveals",
        lambda subtensor, spec: {"hk_z": (submission_z, 5), "hk_a": (submission_a, 5)},
    )
    stub_coordinator(monkeypatch, statuses={
        "r100": make_run_status(file_hashes={"model.gguf": ("sha256", same_sha)}),
    })

    await v.run_stage_1(make_spec())

    winner = store.get_candidate(v._db, "comp1", "hk_a")
    loser = store.get_candidate(v._db, "comp1", "hk_z")
    assert winner["status"] == "done"
    assert loser["status"] == "failed"
    assert "duplicate" in loser["failure_reason"]


@pytest.mark.asyncio
async def test_run_stage_1_retries_when_coordinator_missing_benchmarks(monkeypatch):
    """Coordinator not serving a required benchmark is infra-shaped, not a
    'nobody eligible' outcome — retryable like a chain RPC failure."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    metagraph.neurons = [FakeNeuron("hk1")]
    v = make_validator(monkeypatch, metagraph=metagraph)

    monkeypatch.setattr("validator.chain_scanner.scan_reveals", lambda subtensor, spec: {"hk1": (make_submission(), 5)})
    stub_coordinator(monkeypatch, benchmarks=set())  # missing "mmlu"

    await v.run_stage_1(make_spec())
    assert store.get_stage(v._db, "comp1") == "stage1_ranking"
    assert store.is_scored(v._db, "comp1") is False


# ---------------------------------------------------------------------------
# Scoring inside stage 1 — the work the old stage 2 did per tick
# ---------------------------------------------------------------------------

def _stage1_env(monkeypatch, reveals, statuses=None, benchmarks={"mmlu"}):
    """Drive a full stage 1 from canned reveals and coordinator statuses."""
    monkeypatch.setattr("validator.chain_scanner.scan_reveals", lambda subtensor, spec: reveals)
    stub_coordinator(monkeypatch, statuses=statuses, benchmarks=benchmarks)


@pytest.mark.asyncio
async def test_stage_1_scores_candidate_in_the_same_pass(monkeypatch):
    """There is no second stage: a verified candidate is scored, and the
    competition finalized, within the one stage-1 run."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    _stage1_env(monkeypatch, {"hk1": (make_submission(), 5)})

    await v.run_stage_1(make_spec(top_n=1))

    candidate = store.get_candidate(v._db, "comp1", "hk1")
    assert candidate["status"] == "done", candidate["failure_reason"]
    results = store.scoring_results_for_competition(v._db, "comp1")
    assert len(results) == 1
    # 900 KB is what make_run_status reports as the coordinator's measurement.
    assert results[0]["measured_memory_kb"] == 900
    assert store.get_stage(v._db, "comp1") == "finalized"
    assert store.latest_weights_for_competition(v._db, "comp1") == {"hk1": 1.0}


@pytest.mark.asyncio
async def test_stage_1_fails_candidate_whose_verified_scores_miss_the_floor(monkeypatch):
    """A candidate whose runs were all rejected reaches scoring and is failed
    by the floors, rather than vanishing without a record."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    _stage1_env(monkeypatch, {"hk1": (make_submission(), 5)},
                statuses={"r100": make_run_status(repo="someone/else")})

    await v.run_stage_1(make_spec(top_n=1))

    candidate = store.get_candidate(v._db, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "failed floors" in candidate["failure_reason"]
    assert store.scoring_results_for_competition(v._db, "comp1") == []


@pytest.mark.asyncio
async def test_stage_1_fails_candidate_over_the_memory_cap(monkeypatch):
    """The cap is enforced against the coordinator's measured figure."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    _stage1_env(monkeypatch, {"hk1": (make_submission(), 5)},
                statuses={"r100": make_run_status(measured_memory_bytes=5000 * 1024)})

    await v.run_stage_1(make_spec(
        top_n=1, competition_type="gpu_memory_ceiling", max_gpu_memory_kb=1000,
    ))

    candidate = store.get_candidate(v._db, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "exceeded memory cap" in candidate["failure_reason"]


@pytest.mark.asyncio
async def test_stage_1_backfills_past_a_failing_candidate(monkeypatch):
    """A candidate that fails scoring must not consume a payable slot — the
    next-ranked one takes it, which is the backfill the old stage 2 walked one
    tick at a time."""
    metagraph = FakeMetagraph(hotkeys=["hk_bad", "hk_good"], uids=[0, 1], stake=[1.0, 1.0],
                              weights=[[1.0, 0.0], [0.0, 1.0]], validator_permit=[True, True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    # hk_bad scores higher but reports no memory, so it cannot be scored.
    reveals = {
        "hk_bad": (make_submission(runs=[{"b": "mmlu", "r": "r901"}], file_sha256="b" * 64), 5),
        "hk_good": (make_submission(runs=[{"b": "mmlu", "r": "r902"}]), 5),
    }
    _stage1_env(monkeypatch, reveals, statuses={
        "r901": make_run_status(run_id="r901", scores={"mmlu": 0.9},
                                 file_hashes={"model.gguf": ("sha256", "b" * 64)},
                                 measured_memory_bytes=None),
        "r902": make_run_status(run_id="r902", scores={"mmlu": 0.7}),
    })

    await v.run_stage_1(make_spec(top_n=1))

    assert store.get_candidate(v._db, "comp1", "hk_bad")["status"] == "failed"
    assert store.get_candidate(v._db, "comp1", "hk_good")["status"] == "done"
    assert store.latest_weights_for_competition(v._db, "comp1") == {"hk_good": 1.0}


@pytest.mark.asyncio
async def test_stage_1_stops_scoring_at_top_n(monkeypatch):
    """Only top_n candidates are scored, in rank order — the rest stay standby
    rather than being paid a slot that does not exist."""
    metagraph = FakeMetagraph(hotkeys=["hk_first", "hk_second"], uids=[0, 1], stake=[1.0, 1.0],
                              weights=[[1.0, 0.0], [0.0, 1.0]], validator_permit=[True, True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    reveals = {
        "hk_second": (make_submission(runs=[{"b": "mmlu", "r": "r904"}], file_sha256="b" * 64), 5),
        "hk_first": (make_submission(runs=[{"b": "mmlu", "r": "r903"}]), 5),
    }
    _stage1_env(monkeypatch, reveals, statuses={
        "r903": make_run_status(run_id="r903", scores={"mmlu": 0.9}),
        "r904": make_run_status(run_id="r904", scores={"mmlu": 0.4},
                                 file_hashes={"model.gguf": ("sha256", "b" * 64)}),
    })

    await v.run_stage_1(make_spec(
        top_n=1, competition_type="gpu_memory_ceiling", max_gpu_memory_kb=10_000,
    ))

    assert store.get_candidate(v._db, "comp1", "hk_first")["status"] == "done"
    assert store.get_candidate(v._db, "comp1", "hk_second")["status"] == "standby"


@pytest.mark.asyncio
async def test_stage_1_can_run_twice_without_stranding_the_competition(monkeypatch):
    """Scoring no longer advances a stage marker, so a tick that dies partway
    is re-run from the chain on the next one. The second pass must not trip
    over the rows the first one wrote."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    _stage1_env(monkeypatch, {"hk1": (make_submission(), 5)})

    await v.run_stage_1(make_spec(top_n=1))
    assert store.get_candidate(v._db, "comp1", "hk1")["status"] == "done"

    # Simulate a crash between persisting candidates and finalizing: the
    # candidate rows survive, the scored marker does not.
    v._db.execute("DELETE FROM scored_competitions WHERE competition_id = 'comp1'")
    v._db.commit()
    assert store.is_scored(v._db, "comp1") is False

    await v.run_stage_1(make_spec(top_n=1))

    assert store.get_candidate(v._db, "comp1", "hk1")["status"] == "done"
    assert store.scored_status(v._db, "comp1") == "scored"
    assert store.latest_weights_for_competition(v._db, "comp1") == {"hk1": 1.0}


@pytest.mark.asyncio
async def test_stage_1_scores_a_reveal_committed_under_the_previous_spec(monkeypatch):
    """A reveal is sealed on chain before the validator upgrades. Spec 2 only
    carried an extra `max_memory`, so it must still score."""
    import json as _json

    from common.models.submission import parse_reveal_payload

    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)

    payload = _json.loads(make_submission().model_dump_json())
    payload["spec"] = 2
    payload["max_memory"] = 1000          # the field spec 3 dropped
    old_submission = parse_reveal_payload(_json.dumps(payload))
    assert old_submission is not None, "a spec-2 reveal must still parse"

    _stage1_env(monkeypatch, {"hk1": (old_submission, 5)})
    await v.run_stage_1(make_spec(top_n=1))

    assert store.get_candidate(v._db, "comp1", "hk1")["status"] == "done"
    assert store.latest_weights_for_competition(v._db, "comp1") == {"hk1": 1.0}


@pytest.mark.asyncio
async def test_stage_1_terminal_when_no_candidate_can_be_scored(monkeypatch):
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    _stage1_env(monkeypatch, {"hk1": (make_submission(), 5)},
                statuses={"r100": make_run_status(status="failed", failure_reason="pod died")})

    await v.run_stage_1(make_spec(top_n=1))

    assert store.scored_status(v._db, "comp1") == "failed_no_participants"
    assert store.latest_weights_for_competition(v._db, "comp1") is None


@pytest.mark.asyncio
async def test_leader_loop_finalizes_competition_that_crossed_into_distributing(monkeypatch):
    """A candidate scored on the final scoring tick must still be paid.

    Stage work used to run only in the SCORING phase, so a validator that was
    down across scoring_end_block would never score the competition at all.
    """
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    spec = make_spec(top_n=1, distribution_blocks=1000, reveal_grace_blocks=0)

    monkeypatch.setattr("validator.chain_scanner.scan_reveals",
                        lambda subtensor, _spec: {"hk1": (make_submission(), 5)})
    stub_coordinator(monkeypatch)

    # Block 25 is past scoring_end_block=20 -> DISTRIBUTING, not SCORING.
    from common.models.competition import CompetitionPhase
    assert spec.phase(25) == CompetitionPhase.DISTRIBUTING

    await _run_one_leader_tick(monkeypatch, v, spec, current_block=25)

    assert store.get_stage(v._db, "comp1") == "finalized"
    assert store.latest_weights_for_competition(v._db, "comp1") == {"hk1": 1.0}


@pytest.mark.asyncio
async def test_leader_loop_does_not_rework_an_already_scored_competition(monkeypatch):
    """The is_scored guard must still short-circuit in DISTRIBUTING, so a
    finalized competition is never re-finalized on every tick."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    spec = make_spec(top_n=1, distribution_blocks=1000, reveal_grace_blocks=0)

    store.mark_scored(v._db, "comp1", status="scored")

    called = []
    async def fake_stage_1(s):
        called.append(s.id)
    monkeypatch.setattr(v, "run_stage_1", fake_stage_1)

    await _run_one_leader_tick(monkeypatch, v, spec, current_block=25)
    assert called == []
# ---------------------------------------------------------------------------
# Restart resume — no in-process state, just DB continuity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scoring_state_survives_across_separate_validator_instances(monkeypatch):
    """No background tasks, no in-process state — a 'restart' is just
    constructing a new Validator against the same DB. The verified scores and
    the coordinator's memory figure must both survive, since they are what a
    candidate was scored on."""
    import json

    db_path = Path(tempfile.mkstemp(suffix=".db")[1])
    monkeypatch.setattr(store, "validator_db_path", lambda: db_path)
    monkeypatch.setattr(common_settings, "BITTENSOR", False)
    monkeypatch.setattr(common_settings, "VALIDATOR_MODE", "leader", raising=False)

    v1 = Validator(wallet=FakeWallet(), subtensor=FakeSubtensor(), metagraph=None)
    store.insert_revealed_candidates(v1._db, "comp1", [{
        "hotkey": "hk1", "rank": 0, "submission_json": make_submission().model_dump_json(),
        "reveal_block": 5, "status": "done", "verified_scores": {"mmlu": 0.8},
        "measured_memory_kb": 999,
    }])

    v2 = Validator(wallet=FakeWallet(), subtensor=FakeSubtensor(), metagraph=None)
    candidate = store.get_candidate(v2._db, "comp1", "hk1")
    assert candidate["status"] == "done"
    assert candidate["measured_memory_kb"] == 999
    assert json.loads(candidate["verified_scores_json"]) == {"mmlu": 0.8}


# ---------------------------------------------------------------------------
# follower loop
# ---------------------------------------------------------------------------

async def _run_one_follower_tick(monkeypatch, v, results, scored_status):
    """Drive exactly one iteration of the follower loop, then stop it."""
    class FakeLeaderClient:
        def get_scoring_results(self, competition_id):
            return results, scored_status

    v._leader_client = FakeLeaderClient()

    async def stop(_interval):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        await v._follower_loop()


@pytest.mark.asyncio
async def test_follower_does_not_mark_scored_when_all_weights_zero(monkeypatch):
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    monkeypatch.setattr(validator_settings, "LEADER_VALIDATOR_URL", "http://leader")
    v = make_validator(monkeypatch, metagraph=metagraph, mode="follower")
    spec = make_spec(reveal_grace_blocks=2, scoring_end_block=100)
    monkeypatch.setattr(v, "_get_active_competitions", lambda current_block: [spec])
    monkeypatch.setattr("validator.validator.get_current_block", lambda subtensor: 15)
    monkeypatch.setattr("competition.scoring.compute_emission_weights", lambda ranked, dist: {"hk1": 0.0})

    results = [{"hotkey": "hk1", "competition_id": "comp1", "final_score": 0.7, "measured_memory_kb": 1000}]
    await _run_one_follower_tick(monkeypatch, v, results, "scored")

    assert store.is_scored(v._db, "comp1") is False


@pytest.mark.asyncio
async def test_follower_marks_scored_on_real_weights(monkeypatch):
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    monkeypatch.setattr(validator_settings, "LEADER_VALIDATOR_URL", "http://leader")
    v = make_validator(monkeypatch, metagraph=metagraph, mode="follower")
    spec = make_spec(reveal_grace_blocks=2, scoring_end_block=100)
    monkeypatch.setattr(v, "_get_active_competitions", lambda current_block: [spec])
    monkeypatch.setattr("validator.validator.get_current_block", lambda subtensor: 15)

    results = [{"hotkey": "hk1", "competition_id": "comp1", "final_score": 0.7, "measured_memory_kb": 1000}]
    await _run_one_follower_tick(monkeypatch, v, results, "scored")

    assert store.is_scored(v._db, "comp1") is True
    assert store.latest_weights_for_competition(v._db, "comp1") == {"hk1": 1.0}


# ---------------------------------------------------------------------------
# pause
# ---------------------------------------------------------------------------

async def _run_one_leader_tick(monkeypatch, v, spec, current_block):
    """Drive exactly one _leader_loop iteration.

    The loop is infinite and always sleeps in its finally block, so the sleep
    is what we hijack to break out after the first pass.
    """
    monkeypatch.setattr("validator.validator.get_current_block", lambda subtensor: current_block)
    monkeypatch.setattr(v, "_get_active_competitions", lambda block: [spec])

    async def stop_after_first_tick(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", stop_after_first_tick)
    with pytest.raises(asyncio.CancelledError):
        await v._leader_loop()


@pytest.mark.asyncio
async def test_leader_loop_skips_scoring_while_paused(monkeypatch):
    """The point of the feature: a paused competition gets no stage work."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    spec = make_spec(reveal_grace_blocks=0)

    called = []

    async def fake_stage_1(s):
        called.append(s.id)

    monkeypatch.setattr(v, "run_stage_1", fake_stage_1)
    store.set_paused(v._db, "comp1", True)

    # block 15 -> past commit_end_block=10 and grace, before scoring_end_block=20
    await _run_one_leader_tick(monkeypatch, v, spec, current_block=15)

    assert called == []
    assert store.get_stage(v._db, "comp1") == "stage1_ranking"
    assert store.is_scored(v._db, "comp1") is False


@pytest.mark.asyncio
async def test_leader_loop_runs_scoring_after_resume(monkeypatch):
    """Same tick, same block — only the flag differs."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    spec = make_spec(reveal_grace_blocks=0)

    called = []

    async def fake_stage_1(s):
        called.append(s.id)

    monkeypatch.setattr(v, "run_stage_1", fake_stage_1)
    store.set_paused(v._db, "comp1", True)
    store.set_paused(v._db, "comp1", False)

    await _run_one_leader_tick(monkeypatch, v, spec, current_block=15)

    assert called == ["comp1"]


@pytest.mark.asyncio
async def test_follower_tracks_leader_result_changes_after_finalizing(monkeypatch):
    """The leader's winners can be altered by an admin after it first finalized.
    A follower that latched on the first answer would keep paying the old winner."""
    metagraph = FakeMetagraph(hotkeys=["hk1", "hk2"], uids=[0, 1], stake=[1.0, 1.0],
                              weights=[[1.0], [1.0]], validator_permit=[True, True])
    monkeypatch.setattr(validator_settings, "LEADER_VALIDATOR_URL", "http://leader")
    v = make_validator(monkeypatch, metagraph=metagraph, mode="follower")
    spec = make_spec(reveal_grace_blocks=2, scoring_end_block=100)
    monkeypatch.setattr(v, "_get_active_competitions", lambda current_block: [spec])
    monkeypatch.setattr("validator.validator.get_current_block", lambda subtensor: 15)

    first = [{"hotkey": "hk1", "competition_id": "comp1", "final_score": 0.7, "measured_memory_kb": 1000}]
    await _run_one_follower_tick(monkeypatch, v, first, "scored")
    assert store.latest_weights_for_competition(v._db, "comp1") == {"hk1": 1.0}

    # admin corrects the winner on the leader
    second = [{"hotkey": "hk2", "competition_id": "comp1", "final_score": 0.9, "measured_memory_kb": 1000}]
    await _run_one_follower_tick(monkeypatch, v, second, "scored")
    assert store.latest_weights_for_competition(v._db, "comp1") == {"hk2": 1.0}


@pytest.mark.asyncio
async def test_follower_holds_weights_while_leader_is_mid_reset(monkeypatch):
    """reset-scoring un-finalizes the leader, which then serves []. Dropping our
    weights there would burn the subnet's emissions until it rescores."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    monkeypatch.setattr(validator_settings, "LEADER_VALIDATOR_URL", "http://leader")
    v = make_validator(monkeypatch, metagraph=metagraph, mode="follower")
    spec = make_spec(reveal_grace_blocks=2, scoring_end_block=100)
    monkeypatch.setattr(v, "_get_active_competitions", lambda current_block: [spec])
    monkeypatch.setattr("validator.validator.get_current_block", lambda subtensor: 15)

    results = [{"hotkey": "hk1", "competition_id": "comp1", "final_score": 0.7, "measured_memory_kb": 1000}]
    await _run_one_follower_tick(monkeypatch, v, results, "scored")
    await _run_one_follower_tick(monkeypatch, v, [], None)

    assert store.latest_weights_for_competition(v._db, "comp1") == {"hk1": 1.0}


@pytest.mark.asyncio
async def test_follower_does_not_rewrite_unchanged_weights(monkeypatch):
    """Polling every tick must not append a weights_history row per tick."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    monkeypatch.setattr(validator_settings, "LEADER_VALIDATOR_URL", "http://leader")
    v = make_validator(monkeypatch, metagraph=metagraph, mode="follower")
    spec = make_spec(reveal_grace_blocks=2, scoring_end_block=100)
    monkeypatch.setattr(v, "_get_active_competitions", lambda current_block: [spec])
    monkeypatch.setattr("validator.validator.get_current_block", lambda subtensor: 15)

    results = [{"hotkey": "hk1", "competition_id": "comp1", "final_score": 0.7, "measured_memory_kb": 1000}]
    for _ in range(3):
        await _run_one_follower_tick(monkeypatch, v, results, "scored")

    assert len(store.weights_history_for_competition(v._db, "comp1")) == 1
