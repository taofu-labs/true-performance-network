import pytest

from common import settings as common_settings
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
