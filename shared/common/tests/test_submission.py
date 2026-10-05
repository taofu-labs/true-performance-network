"""
Payload model — the on-chain commit shape.

Miners commit coordinator run ids rather than self-reported scores, under
payload spec version 2. A spec-1 payload (or anything still carrying `claims`)
must be rejected outright rather than silently scoring zero.
"""
import json

import pytest
from pydantic import ValidationError

from common.models.submission import (
    PAYLOAD_SPEC_VERSION,
    BenchmarkRun,
    MinerSubmission,
    build_reveal_payload,
    parse_reveal_payload,
)


def make_runs():
    return [BenchmarkRun(b="mmlu", r="r1234")]


def make_payload(**overrides) -> str:
    fields = dict(
        competition_id="comp1",
        repository="user/repo",
        file="model.gguf",
        file_sha256="a" * 64,
        max_memory=1000,
        runs=make_runs(),
        huggingface_revision="b" * 40,
    )
    fields.update(overrides)
    return build_reveal_payload(**fields)


# ---------------------------------------------------------------------------
# Run id shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "",                    # empty
    "r0",                  # short ids never start at zero
    "r01234",              # nor carry a leading zero
    "1234",                # missing the r prefix
    "notarunid",
    "r12 34",              # whitespace
    "../../etc/passwd",    # path traversal shaped
    "4f4e3548-32b9-2120",  # truncated uuid
])
def test_rejects_malformed_run_ids(bad):
    with pytest.raises(ValidationError):
        BenchmarkRun(b="mmlu", r=bad)


# ---------------------------------------------------------------------------
# Payload spec version
# ---------------------------------------------------------------------------

def test_build_reveal_payload_stamps_current_spec_version():
    assert json.loads(make_payload())["spec"] == PAYLOAD_SPEC_VERSION == 2


def test_round_trips_through_parse():
    submission = parse_reveal_payload(make_payload())
    assert submission is not None
    assert submission.repository == "user/repo"
    assert submission.run_ids == {"mmlu": "r1234"}


def test_rejects_spec_1_payload():
    """An old-build miner committing against a new-build validator must fail
    loudly at reveal, not silently score zero on an unrecognised field."""
    payload = make_payload().replace('"spec":2', '"spec":1')
    assert parse_reveal_payload(payload) is None


def test_rejects_payload_carrying_claims_instead_of_runs():
    payload = json.dumps({
        "spec": 2, "competition_id": "comp1", "repository": "user/repo",
        "file": "model.gguf", "file_sha256": "a" * 64, "max_memory": 1000,
        "huggingface_revision": "b" * 40,
        "claims": [{"b": "mmlu", "s": 0.9}],
    })
    assert parse_reveal_payload(payload) is None


def test_rejects_malformed_json():
    assert parse_reveal_payload("{not json") is None


# ---------------------------------------------------------------------------
# run_ids accessor
# ---------------------------------------------------------------------------

def test_run_ids_maps_benchmark_to_run():
    submission = MinerSubmission(
        competition_id="comp1",
        runs=[BenchmarkRun(b="mmlu", r="r1"), BenchmarkRun(b="gsm8k", r="r2")],
        repository="user/repo", file="m.gguf", file_sha256="a" * 64,
        max_memory=1000, huggingface_revision="b" * 40,
    )
    assert submission.run_ids == {"mmlu": "r1", "gsm8k": "r2"}


def test_run_ids_duplicate_benchmark_keeps_last():
    """A miner listing the same benchmark twice gets a deterministic outcome
    rather than an error — the later id wins."""
    submission = MinerSubmission(
        competition_id="comp1",
        runs=[BenchmarkRun(b="mmlu", r="r1"), BenchmarkRun(b="mmlu", r="r2")],
        repository="user/repo", file="m.gguf", file_sha256="a" * 64,
        max_memory=1000, huggingface_revision="b" * 40,
    )
    assert submission.run_ids == {"mmlu": "r2"}


def test_empty_runs_list_is_allowed_and_scores_nothing():
    """Parsing must succeed — the candidate is then scored 0.0 per benchmark
    during verification, rather than being dropped without a record."""
    submission = parse_reveal_payload(make_payload(runs=[]))
    assert submission is not None
    assert submission.run_ids == {}


# ---------------------------------------------------------------------------
# Field validation retained from the previous payload shape
# ---------------------------------------------------------------------------

def test_rejects_repository_as_url():
    assert parse_reveal_payload(make_payload(repository="https://huggingface.co/user/repo")) is None


def test_normalises_revision_case():
    submission = parse_reveal_payload(make_payload(huggingface_revision="B" * 40))
    assert submission.huggingface_revision == "b" * 40


def test_payload_is_minified_for_chain():
    """Payload size matters — it is TLE-encrypted into a chain commitment."""
    payload = make_payload()
    assert ", " not in payload and '": ' not in payload
