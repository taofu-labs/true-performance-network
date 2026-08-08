import pytest

from common import settings as common_settings
from common.models.competition import BenchmarkTask, CompetitionSpec
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


@pytest.mark.asyncio
async def test_run_scoring_retries_instead_of_terminal_when_no_reveals(monkeypatch):
    """C11-C14 finding #1: no reveals is retryable, not an immediate terminal
    success — a transient scan failure must not mark the competition scored."""
    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)
    monkeypatch.setattr("validator.chain_scanner.scan_reveals", lambda subtensor, spec, db: {})

    reason = await v._run_scoring(make_spec(), current_block=15)
    assert reason == "no_reveals"


@pytest.mark.asyncio
async def test_run_scoring_retries_when_nothing_passes_precheck(monkeypatch):
    """C11-C14 finding #1: precheck container startup failure (or every
    candidate failing precheck) must not be scored as a terminal success with
    zero participants — it's validator-side infra failure, not participant
    failure."""
    from common.models.submission import Claim, MinerSubmission

    class FakeNeuron:
        def __init__(self, hotkey):
            self.hotkey = hotkey
            self.collateral_locked = None

    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    metagraph.neurons = [FakeNeuron("hk1")]
    v = make_validator(monkeypatch, metagraph=metagraph)

    submission = MinerSubmission(
        competition_id="comp1", claims=[Claim(b="mmlu", s=0.7)], repository="user/repo",
        file="model.gguf", file_sha256="a" * 64, max_memory=1000, huggingface_revision="a" * 40,
    )
    monkeypatch.setattr(
        "validator.chain_scanner.scan_reveals",
        lambda subtensor, spec, db: {"hk1": (submission, 5)},
    )
    monkeypatch.setattr("competition.benchmark_client.make_coordinator", lambda: type(
        "C", (), {"list_benchmarks": lambda self: {"mmlu"}}
    )())
    monkeypatch.setattr(
        "competition.precheck_client.PrecheckContainer.start",
        lambda self: (_ for _ in ()).throw(RuntimeError("docker daemon unavailable")),
    )

    reason = await v._run_scoring(make_spec(), current_block=15)
    assert reason == "no_participants"
    assert store_module_is_scored_false(v)


def store_module_is_scored_false(v) -> bool:
    from validator import store
    return store.is_scored(v._db, "comp1") is False


@pytest.mark.asyncio
async def test_reconciliation_marks_scored_when_scoring_run_committed_but_status_missing(monkeypatch):
    """Autoupdate can SIGKILL the validator between record_scoring_run()
    committing and the follow-up mark_scored() call — the next tick must
    detect and reconcile that drift rather than re-scoring or hanging
    forever."""
    from validator import store

    metagraph = FakeMetagraph(hotkeys=["hk1"], uids=[0], stake=[1.0], weights=[[1.0]], validator_permit=[True])
    v = make_validator(monkeypatch, metagraph=metagraph)

    store.record_scoring_run(v._db, "comp1", block=100, outcomes=[], reveals={})
    assert store.is_scored(v._db, "comp1") is False

    if not store.is_scored(v._db, "comp1") and store.has_scoring_run(v._db, "comp1"):
        store.mark_scored(v._db, "comp1")

    assert store.is_scored(v._db, "comp1") is True
