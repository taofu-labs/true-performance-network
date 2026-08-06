import tenacity

from common import chain, settings as common_settings


class FakeResult:
    def __init__(self, success: bool):
        self.success = success


class FakeWallet:
    def __init__(self, ss58: str):
        class _Hotkey:
            ss58_address = ss58
        self.hotkey = _Hotkey()


class FakeCommitment:
    def __init__(self, hotkey, revealed):
        self.hotkey = hotkey
        self.revealed = revealed


class FakeMetagraph:
    def __init__(self, commitments):
        self.commitments = commitments


class FakeNeurons:
    def __init__(self, registered_uid):
        self._registered_uid = registered_uid

    def uid(self, hotkey_ss58, netuid):
        return self._registered_uid


class FakeSubnets:
    def __init__(self, metagraph):
        self._metagraph = metagraph

    def metagraph(self, netuid):
        return self._metagraph


class FakeSubtensor:
    def __init__(self, submit_success=True, registered_uid=1, commitments=None, existing_commitment=None):
        self._submit_success = submit_success
        self.neurons = FakeNeurons(registered_uid)
        self.subnets = FakeSubnets(FakeMetagraph(commitments or {}))
        self.policy = None
        self._existing_commitment = existing_commitment

    def submit_call(self, call, wallet, signer=None):
        return FakeResult(self._submit_success)

    def query(self, item, params=None):
        return self._existing_commitment


def test_current_block_calls_when_callable():
    class OldSubtensor:
        def block(self):
            return 123
    assert chain.current_block(OldSubtensor()) == 123


def test_current_block_reads_when_property():
    class NewSubtensor:
        block = 456
    assert chain.current_block(NewSubtensor()) == 456


def test_get_subtensor_raises_when_bittensor_disabled(monkeypatch):
    monkeypatch.setattr(common_settings, "BITTENSOR", False)
    monkeypatch.setattr(chain.get_subtensor.retry, "stop", tenacity.stop_after_attempt(1))
    try:
        chain.get_subtensor()
        assert False, "expected an exception"
    except tenacity.RetryError as e:
        assert "BITTENSOR=False" in str(e.last_attempt.exception())


def test_timelocked_commit_returns_result_success():
    wallet = FakeWallet("5FakeHotkey")
    subtensor = FakeSubtensor(submit_success=True)
    result = chain.timelocked_commit(subtensor, wallet, netuid=2, reveal_payload="{}", blocks_until_reveal=10)
    assert result.success is True
    assert subtensor.policy is None  # restored after the call


def test_timelocked_commit_returns_false_on_failure():
    wallet = FakeWallet("5FakeHotkey")
    subtensor = FakeSubtensor(submit_success=False)
    result = chain.timelocked_commit(subtensor, wallet, netuid=2, reveal_payload="{}", blocks_until_reveal=10)
    assert result.success is False


def test_timelocked_commit_swallows_exceptions():
    class RaisingSubtensor(FakeSubtensor):
        def submit_call(self, call, wallet, signer=None):
            raise RuntimeError("chain unreachable")

    wallet = FakeWallet("5FakeHotkey")
    subtensor = RaisingSubtensor()
    result = chain.timelocked_commit(subtensor, wallet, netuid=2, reveal_payload="{}", blocks_until_reveal=10)
    assert result.success is False
    assert subtensor.policy is None  # restored even on failure


def test_timelocked_commit_appends_to_existing_pending_field():
    wallet = FakeWallet("5FakeHotkey")
    existing = {"info": {"fields": [{"TimelockEncrypted": {"encrypted": [1, 2, 3], "reveal_round": 111}}]}}
    subtensor = FakeSubtensor(submit_success=True, existing_commitment=existing)
    captured = {}

    from bittensor import calls as calls_mod
    original_set_commitment = calls_mod.Commitments.set_commitment

    def capture_set_commitment(netuid, info):
        captured["info"] = info
        return original_set_commitment(netuid, info)

    import unittest.mock
    with unittest.mock.patch.object(calls_mod.Commitments, "set_commitment", staticmethod(capture_set_commitment)):
        result = chain.timelocked_commit(subtensor, wallet, netuid=2, reveal_payload="{}", blocks_until_reveal=10)

    assert result.success is True
    assert len(captured["info"]["fields"]) == 2
    assert captured["info"]["fields"][0]["TimelockEncrypted"]["reveal_round"] == 111


def test_timelocked_commit_raises_when_max_fields_reached():
    wallet = FakeWallet("5FakeHotkey")
    existing = {
        "info": {
            "fields": [
                {"TimelockEncrypted": {"encrypted": [1], "reveal_round": 111}},
                {"TimelockEncrypted": {"encrypted": [2], "reveal_round": 222}},
                {"TimelockEncrypted": {"encrypted": [3], "reveal_round": 333}},
            ]
        }
    }
    subtensor = FakeSubtensor(submit_success=True, existing_commitment=existing)
    wallet2 = FakeWallet("5FakeHotkey")
    try:
        chain.timelocked_commit(subtensor, wallet2, netuid=2, reveal_payload="{}", blocks_until_reveal=10)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "max 3" in str(e) or "3 pending" in str(e)


def test_timelocked_commit_replaces_matching_reveal_round():
    wallet = FakeWallet("5FakeHotkey")
    existing = {
        "info": {
            "fields": [
                {"TimelockEncrypted": {"encrypted": [9, 9, 9], "reveal_round": 555}},
            ]
        }
    }
    subtensor = FakeSubtensor(submit_success=True, existing_commitment=existing)
    captured = {}

    from bittensor import calls as calls_mod
    original_set_commitment = calls_mod.Commitments.set_commitment

    def capture_set_commitment(netuid, info):
        captured["info"] = info
        return original_set_commitment(netuid, info)

    import unittest.mock
    with unittest.mock.patch.object(calls_mod.Commitments, "set_commitment", staticmethod(capture_set_commitment)):
        result = chain.timelocked_commit(
            subtensor, wallet, netuid=2, reveal_payload="{}", blocks_until_reveal=10,
            previous_reveal_round=555,
        )

    assert result.success is True
    assert len(captured["info"]["fields"]) == 1
    assert captured["info"]["fields"][0]["TimelockEncrypted"]["encrypted"] != [9, 9, 9]


def test_is_hotkey_registered_true_and_false():
    assert chain.is_hotkey_registered(FakeSubtensor(registered_uid=1), "hk", netuid=2) is True
    assert chain.is_hotkey_registered(FakeSubtensor(registered_uid=None), "hk", netuid=2) is False


def test_is_hotkey_registered_swallows_exceptions():
    class RaisingNeurons:
        def uid(self, hotkey_ss58, netuid):
            raise RuntimeError("chain unreachable")

    class RaisingSubtensor:
        neurons = RaisingNeurons()

    assert chain.is_hotkey_registered(RaisingSubtensor(), "hk", netuid=2) is False


def test_read_revealed_commitments_groups_by_hotkey():
    commitments = {1: FakeCommitment("hk1", [(42, '{"a":1}')])}
    subtensor = FakeSubtensor(commitments=commitments)
    result = chain.read_revealed_commitments(subtensor, netuid=2)
    assert result == {"hk1": [('{"a":1}', 42)]}


def test_read_revealed_commitments_skips_unrevealed():
    commitments = {1: FakeCommitment("hk1", [])}
    subtensor = FakeSubtensor(commitments=commitments)
    assert chain.read_revealed_commitments(subtensor, netuid=2) == {}


def test_read_revealed_commitments_swallows_exceptions():
    class RaisingSubnets:
        def metagraph(self, netuid):
            raise RuntimeError("boom")

    class RaisingSubtensor:
        subnets = RaisingSubnets()

    assert chain.read_revealed_commitments(RaisingSubtensor(), netuid=2) == {}


def test_read_revealed_commitments_returns_empty_when_metagraph_missing():
    class NoneSubnets:
        def metagraph(self, netuid):
            return None

    class NoneSubtensor:
        subnets = NoneSubnets()

    assert chain.read_revealed_commitments(NoneSubtensor(), netuid=2) == {}
