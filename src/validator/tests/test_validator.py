import tempfile
from pathlib import Path

import pytest

from common import settings as common_settings
from common.models.competition import BenchmarkTask, CompetitionSpec
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


def test_construct_without_bittensor_does_not_touch_network(monkeypatch):
    v = make_validator(monkeypatch)
    assert v.subtensor is not None
    assert v.metagraph is None  # BITTENSOR=False -> never auto-created, none injected


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


def test_copy_weights_from_chain_no_uids_returns_empty(monkeypatch):
    metagraph = FakeMetagraph(hotkeys=[], uids=[], stake=[], weights=[], validator_permit=[])
    v = make_validator(monkeypatch, metagraph=metagraph)
    assert v.copy_weights_from_chain() == {}


@pytest.mark.asyncio
async def test_set_weights_skips_when_bittensor_disabled(monkeypatch):
    v = make_validator(monkeypatch)
    monkeypatch.setattr(common_settings, "BITTENSOR", False)
    await v.set_weights(weights={0: 1.0})  # must not raise, no wallet/subtensor calls needed


@pytest.mark.asyncio
async def test_set_weights_skips_when_all_scores_zero(monkeypatch):
    metagraph = FakeMetagraph(
        hotkeys=["hk0"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True],
    )
    v = make_validator(monkeypatch, metagraph=metagraph)
    monkeypatch.setattr(common_settings, "BITTENSOR", True)
    calls = []
    monkeypatch.setattr("bittensor.set_weights", lambda *a, **kwargs: calls.append(kwargs))
    await v.set_weights(weights={0: 0.0, 1: 0.0})
    assert calls == []


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
        "benchmarks": [BenchmarkTask(name="mmlu", min_score=0.5, weight=1.0)],
    }
    fields.update(overrides)
    return CompetitionSpec.model_validate(fields)


class FakeNeuron:
    def __init__(self, hotkey):
        self.hotkey = hotkey
        self.collateral_locked = None


def make_submission(**overrides):
    from common.models.submission import Claim, MinerSubmission
    fields = dict(
        competition_id="comp1", claims=[Claim(b="mmlu", s=0.7)], repository="user/repo",
        file="model.gguf", file_sha256="a" * 64, max_memory=1000, huggingface_revision="a" * 40,
    )
    fields.update(overrides)
    return MinerSubmission(**fields)


# ---------------------------------------------------------------------------
# run_stage_1
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_stage_1_terminal_when_no_reveals(monkeypatch):
    """No reveals is a terminal, non-retryable outcome — reveals are
    chain-provably immutable once the commit window closes."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    monkeypatch.setattr("validator.chain_scanner.scan_reveals", lambda subtensor, spec, db: {})

    await v.run_stage_1(make_spec())
    assert store.scored_status(v._db, "comp1") == "failed_no_reveals"
    assert store.is_scored(v._db, "comp1") is True


@pytest.mark.asyncio
async def test_run_stage_1_retries_on_infra_failure(monkeypatch):
    """A chain-read exception (not an empty result) must retry, capped at
    STAGE1_MAX_ATTEMPTS, distinct from the terminal no-reveals path."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)

    def boom(subtensor, spec, db):
        raise RuntimeError("chain RPC failed")
    monkeypatch.setattr("validator.chain_scanner.scan_reveals", boom)

    await v.run_stage_1(make_spec())
    assert store.get_stage(v._db, "comp1") == "stage1_ranking"  # not terminal yet
    assert store.is_scored(v._db, "comp1") is False

    await v.run_stage_1(make_spec())
    await v.run_stage_1(make_spec())
    assert store.scored_status(v._db, "comp1") == "failed_stage1_infra"


@pytest.mark.asyncio
async def test_run_stage_1_persists_ranked_candidates_and_advances_stage(monkeypatch):
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    metagraph.neurons = [FakeNeuron("hk1")]
    v = make_validator(monkeypatch, metagraph=metagraph)

    submission = make_submission()
    monkeypatch.setattr("validator.chain_scanner.scan_reveals", lambda subtensor, spec, db: {"hk1": (submission, 5)})
    monkeypatch.setattr("competition.benchmark_client.make_coordinator", lambda: type(
        "C", (), {"list_benchmarks": lambda self: {"mmlu"}}
    )())

    await v.run_stage_1(make_spec())

    assert store.get_stage(v._db, "comp1") == "stage2_scoring"
    candidate = store.get_candidate(v._db, "comp1", "hk1")
    assert candidate["status"] == "standby"
    assert candidate["rank"] == 0


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
        lambda subtensor, spec, db: {"hk_z": (submission_z, 5), "hk_a": (submission_a, 5)},
    )
    monkeypatch.setattr("competition.benchmark_client.make_coordinator", lambda: type(
        "C", (), {"list_benchmarks": lambda self: {"mmlu"}}
    )())

    await v.run_stage_1(make_spec())

    winner = store.get_candidate(v._db, "comp1", "hk_a")
    loser = store.get_candidate(v._db, "comp1", "hk_z")
    assert winner["status"] == "standby"
    assert loser["status"] == "failed"
    assert "duplicate" in loser["failure_reason"]


@pytest.mark.asyncio
async def test_run_stage_1_retries_when_coordinator_missing_benchmarks(monkeypatch):
    """Coordinator not serving a required benchmark is infra-shaped, not a
    'nobody eligible' outcome — retryable like a chain RPC failure."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    metagraph.neurons = [FakeNeuron("hk1")]
    v = make_validator(monkeypatch, metagraph=metagraph)

    submission = make_submission()
    monkeypatch.setattr("validator.chain_scanner.scan_reveals", lambda subtensor, spec, db: {"hk1": (submission, 5)})
    monkeypatch.setattr("competition.benchmark_client.make_coordinator", lambda: type(
        "C", (), {"list_benchmarks": lambda self: set()}  # missing "mmlu"
    )())

    await v.run_stage_1(make_spec())
    assert store.get_stage(v._db, "comp1") == "stage1_ranking"
    assert store.is_scored(v._db, "comp1") is False


# ---------------------------------------------------------------------------
# run_stage_2
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_stage_2_finalizes_when_scoring_window_closed(monkeypatch):
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    store.set_stage(v._db, "comp1", "stage2_scoring")
    monkeypatch.setattr("competition.precheck_client.stop_container", lambda cid: None)

    spec = make_spec()  # scoring_end_block=20
    await v.run_stage_2(spec, current_block=25)  # window closed

    assert store.get_stage(v._db, "comp1") == "finalized"
    assert store.is_scored(v._db, "comp1") is True


@pytest.mark.asyncio
async def test_run_stage_2_finalizes_when_top_n_reached(monkeypatch):
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    store.set_stage(v._db, "comp1", "stage2_scoring")
    store.insert_revealed_candidate(v._db, "comp1", "hk1", rank=0, submission_json=make_submission().model_dump_json(), reveal_block=5, status="done")
    v._db.commit()
    store.record_scoring_result(v._db, "comp1", "hk1", final_score=0.7, max_memory_kb=1000)
    v._db.commit()
    monkeypatch.setattr("competition.precheck_client.stop_container", lambda cid: None)

    spec = make_spec(top_n=1)
    await v.run_stage_2(spec, current_block=15)

    assert store.get_stage(v._db, "comp1") == "finalized"
    weights = store.latest_weights_for_competition(v._db, "comp1")
    assert weights == {"hk1": 1.0}


@pytest.mark.asyncio
async def test_run_stage_2_finalizes_when_candidates_exhausted(monkeypatch):
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    store.set_stage(v._db, "comp1", "stage2_scoring")
    store.insert_revealed_candidate(v._db, "comp1", "hk1", rank=0, submission_json=make_submission().model_dump_json(), reveal_block=5, status="failed", failure_reason="x")
    v._db.commit()
    monkeypatch.setattr("competition.precheck_client.stop_container", lambda cid: None)

    # more payable slots than candidates that could ever fill them
    spec = make_spec(top_n=5, emission_distribution=[0.4, 0.3, 0.15, 0.1, 0.05])
    await v.run_stage_2(spec, current_block=15)

    assert store.get_stage(v._db, "comp1") == "finalized"


@pytest.mark.asyncio
async def test_run_stage_2_prechecks_standby_candidates(monkeypatch):
    """A standby candidate should be pulled through precheck and land in
    'queued' on precheck pass — the queue-feeding half of one stage-2 pass."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    store.set_stage(v._db, "comp1", "stage2_scoring")
    store.insert_revealed_candidate(v._db, "comp1", "hk1", rank=0, submission_json=make_submission().model_dump_json(), reveal_block=5, status="standby")
    v._db.commit()

    monkeypatch.setattr("competition.benchmark_client.make_coordinator", lambda: type(
        "C", (), {"poll": lambda self, run_id: None}
    )())
    monkeypatch.setattr("competition.precheck_client.PrecheckContainer.launch", lambda self: None)
    monkeypatch.setattr("competition.precheck_client.is_container_up", lambda cid: True)

    from validator.scorer import PrecheckResult
    monkeypatch.setattr(
        "validator.scorer.precheck_one",
        lambda hotkey, submission, spec, ctr, conn: PrecheckResult(True, gguf_file="model.gguf", measured_memory_kb=999),
    )

    spec = make_spec(top_n=1)
    await v.run_stage_2(spec, current_block=15)

    candidate = store.get_candidate(v._db, "comp1", "hk1")
    assert candidate["status"] == "queued"
    assert candidate["gguf_file"] == "model.gguf"


@pytest.mark.asyncio
async def test_run_stage_2_backfills_on_precheck_failure(monkeypatch):
    """A failed precheck must not stop the whole competition — the next
    standby is tried on the same pass since queue_slots stays open."""
    metagraph = FakeMetagraph(hotkeys=["hk1", "hk2"], uids=[0, 1], stake=[1.0, 1.0], weights=[[1.0, 0.0], [0.0, 1.0]], validator_permit=[True, True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    store.set_stage(v._db, "comp1", "stage2_scoring")
    store.insert_revealed_candidate(v._db, "comp1", "hk1", rank=0, submission_json=make_submission().model_dump_json(), reveal_block=5, status="standby")
    store.insert_revealed_candidate(v._db, "comp1", "hk2", rank=1, submission_json=make_submission().model_dump_json(), reveal_block=5, status="standby")
    v._db.commit()

    monkeypatch.setattr("competition.benchmark_client.make_coordinator", lambda: type(
        "C", (), {"poll": lambda self, run_id: None}
    )())
    monkeypatch.setattr("competition.precheck_client.PrecheckContainer.launch", lambda self: None)
    monkeypatch.setattr("competition.precheck_client.is_container_up", lambda cid: True)

    from validator.scorer import PrecheckResult

    def fake_precheck(hotkey, submission, spec, ctr, conn):
        if hotkey == "hk1":
            return PrecheckResult(False, reason="provenance fail")
        return PrecheckResult(True, gguf_file="model.gguf", measured_memory_kb=999)
    monkeypatch.setattr("validator.scorer.precheck_one", fake_precheck)

    spec = make_spec(top_n=1)
    await v.run_stage_2(spec, current_block=15)

    assert store.get_candidate(v._db, "comp1", "hk1")["status"] == "failed"
    assert store.get_candidate(v._db, "comp1", "hk2")["status"] == "queued"


@pytest.mark.asyncio
async def test_run_stage_2_skips_precheck_without_blocking_when_container_not_ready(monkeypatch):
    """Regression guard for the fix: launch() must be called (fast,
    non-blocking) but precheck work must be skipped entirely — not waited
    on — while the container is still starting (e.g. downloading a base
    model). Benchmark polling/submit for already-queued/benchmarking
    candidates must still happen this tick regardless."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    store.set_stage(v._db, "comp1", "stage2_scoring")
    store.insert_revealed_candidate(v._db, "comp1", "hk1", rank=0, submission_json=make_submission().model_dump_json(), reveal_block=5, status="standby")
    v._db.commit()

    monkeypatch.setattr("competition.benchmark_client.make_coordinator", lambda: type(
        "C", (), {"poll": lambda self, run_id: None}
    )())

    launch_calls = {"n": 0}
    monkeypatch.setattr("competition.precheck_client.PrecheckContainer.launch", lambda self: launch_calls.__setitem__("n", launch_calls["n"] + 1))
    monkeypatch.setattr("competition.precheck_client.is_container_up", lambda cid: False)  # still starting

    def exploding_precheck(*a, **k):
        raise AssertionError("precheck_one must not be called while container is not ready")
    monkeypatch.setattr("validator.scorer.precheck_one", exploding_precheck)

    spec = make_spec(top_n=1)
    await v.run_stage_2(spec, current_block=15)

    assert launch_calls["n"] == 1  # launch() was still called (idempotent, cheap)
    candidate = store.get_candidate(v._db, "comp1", "hk1")
    assert candidate["status"] == "standby"  # untouched — not pulled into precheck this tick


# ---------------------------------------------------------------------------
# Restart resume — no in-process state, just DB continuity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stage2_scoring_state_survives_across_separate_validator_instances(monkeypatch):
    """No background tasks, no in-process container tracking — a 'restart'
    is just constructing a new Validator against the same DB and picking up
    exactly where the queue state left off."""
    db_path = Path(tempfile.mkstemp(suffix=".db")[1])
    monkeypatch.setattr(store, "validator_db_path", lambda: db_path)
    monkeypatch.setattr(common_settings, "BITTENSOR", False)
    monkeypatch.setattr(common_settings, "VALIDATOR_MODE", "leader", raising=False)

    v1 = Validator(wallet=FakeWallet(), subtensor=FakeSubtensor(), metagraph=None)
    store.set_stage(v1._db, "comp1", "stage2_scoring")
    store.insert_revealed_candidate(v1._db, "comp1", "hk1", rank=0, submission_json=make_submission().model_dump_json(), reveal_block=5, status="standby")
    v1._db.commit()
    store.mark_precheck_passed(v1._db, "comp1", "hk1", gguf_file="model.gguf", measured_memory_kb=999)

    v2 = Validator(wallet=FakeWallet(), subtensor=FakeSubtensor(), metagraph=None)
    candidate = store.get_candidate(v2._db, "comp1", "hk1")
    assert candidate["status"] == "queued"
    assert candidate["gguf_file"] == "model.gguf"
