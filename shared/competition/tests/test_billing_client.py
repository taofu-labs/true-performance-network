"""
Miner-side billing API client.

Miners pay for their own benchmark runs through the benchmark service's billing
API. The run id this returns is what goes on chain, so the important behaviours
are: the right value is extracted, and money-related failures surface clearly
rather than being swallowed.
"""
import pytest

from competition.billing_client import (
    BillingClient,
    BillingError,
    Order,
    build_request,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if self._payload is _NOT_JSON:
            raise ValueError("not json")
        return self._payload


_NOT_JSON = object()


def make_client(responses, monkeypatch) -> BillingClient:
    """Client whose every request returns the next canned response.

    `responses` maps "METHOD /path" to a FakeResponse, or is a single
    FakeResponse used for everything.
    """
    client = BillingClient(base_url="http://billing.invalid/api/v1", api_key="bapi_test")
    calls = []

    def fake_request(method, url, **kwargs):
        path = url.replace("http://billing.invalid/api/v1", "")
        calls.append((method, path, kwargs.get("json")))
        if isinstance(responses, dict):
            return responses[f"{method} {path}"]
        return responses

    client._session.request = fake_request
    client.calls = calls
    return client


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_requires_an_api_key():
    with pytest.raises(BillingError) as exc:
        BillingClient(base_url="http://billing.invalid/api/v1", api_key="")
    assert exc.value.code == "missing_api_key"


# ---------------------------------------------------------------------------
# Quote
# ---------------------------------------------------------------------------

def test_quote_extracts_price_and_confirmation_token(monkeypatch):
    client = make_client(FakeResponse({
        "quote": {"estimated_runtime_seconds": 600},
        "billing_price": {
            "charged_cents": 650, "payable_cents": 650, "currency": "usd",
            "review_required": False, "available_credit_cents": 5000,
            "missing_credit_cents": 0,
        },
        "reuse": {"eligible": False},
        "quote_confirmation_token": "signed-token",
    }), monkeypatch)

    quote = client.quote(build_request("user/repo", "a" * 40, "m.gguf", "mmlu"))
    assert quote.charged_cents == 650
    assert quote.available_credit_cents == 5000
    assert quote.confirmation_token == "signed-token"
    assert quote.estimated_runtime_seconds == 600


def test_quote_reports_missing_credit(monkeypatch):
    client = make_client(FakeResponse({
        "billing_price": {
            "charged_cents": 900, "payable_cents": 900, "currency": "usd",
            "available_credit_cents": 100, "missing_credit_cents": 800,
            "review_required": False,
        },
        "quote_confirmation_token": "t",
    }), monkeypatch)

    quote = client.quote({"benchmark": "mmlu"})
    assert quote.missing_credit_cents == 800


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def test_submit_echoes_confirmation_token_and_asserts_price(monkeypatch):
    """The signed token binds the price the miner was shown; sending the
    expected charge too means a price change is rejected, not silently paid."""
    from competition.billing_client import Quote

    client = make_client(FakeResponse({"order": {"id": "o1", "status": "queueing", "charged_cents": 650}}), monkeypatch)
    quote = Quote(charged_cents=650, payable_cents=650, available_credit_cents=5000,
                  missing_credit_cents=0, currency="usd", review_required=False,
                  confirmation_token="signed-token")

    order = client.submit({"benchmark": "mmlu"}, quote)
    assert order.id == "o1"

    _, _, body = client.calls[-1]
    assert body["quote_confirmation_token"] == "signed-token"
    assert body["expected_charged_cents"] == 650
    assert "force" not in body


def test_submit_without_a_quote_forces_current_price(monkeypatch):
    client = make_client(FakeResponse({"order": {"id": "o1", "status": "queueing"}}), monkeypatch)

    client.submit({"benchmark": "mmlu"})
    _, _, body = client.calls[-1]
    assert body["force"] is True
    assert "quote_confirmation_token" not in body


def test_submit_surfaces_insufficient_credit(monkeypatch):
    """402 is the common real-world failure and must be actionable, not a
    generic HTTP error."""
    client = make_client(FakeResponse({"error": {
        "code": "insufficient_credit",
        "message": "Not enough prepaid credit.",
        "details": {"top_up_url": "https://billing/topup"},
    }}, status_code=402), monkeypatch)

    with pytest.raises(BillingError) as exc:
        client.submit({"benchmark": "mmlu"})
    assert exc.value.code == "insufficient_credit"
    assert exc.value.status_code == 402
    assert "Not enough prepaid credit" in str(exc.value)


def test_unreachable_service_is_reported_as_such():
    import requests

    client = BillingClient(base_url="http://billing.invalid/api/v1", api_key="k")

    def boom(*a, **k):
        raise requests.ConnectionError("no route to host")
    client._session.request = boom

    with pytest.raises(BillingError) as exc:
        client.capabilities()
    assert exc.value.code == "unreachable"


# ---------------------------------------------------------------------------
# Waiting for the run id — the value that goes on chain
# ---------------------------------------------------------------------------

def test_order_status_reads_run_id_from_coordinator_status(monkeypatch):
    """The billing order id is NOT the run id; the coordinator's id arrives
    in a separate block of the status response."""
    client = make_client(FakeResponse({
        "order": {"id": "o1", "status": "queued", "charged_cents": 650},
        "coordinator_status": {"status": "queued", "run_id": "uuid-1", "short_id": "r1234"},
    }), monkeypatch)

    order = client.order_status("o1")
    assert order.id == "o1"
    assert order.run_id == "uuid-1"
    assert order.short_id == "r1234"


def test_commit_id_prefers_the_short_id():
    """Short ids cost far less in a TLE-encrypted chain commitment."""
    assert Order(id="o", status="queued", run_id="uuid-1", short_id="r5").commit_id == "r5"


def test_wait_for_run_id_polls_until_the_id_appears(monkeypatch):
    """The id appears when the coordinator accepts the work, well before the
    benchmark finishes — waiting for a score would block for hours."""
    monkeypatch.setattr("competition.billing_client.time.sleep", lambda s: None)
    client = BillingClient(base_url="http://billing.invalid/api/v1", api_key="k")

    polls = {"n": 0}

    def fake_status(order_id):
        polls["n"] += 1
        if polls["n"] < 3:
            return Order(id=order_id, status="queueing")
        return Order(id=order_id, status="queued", short_id="r777")

    client.order_status = fake_status
    order = client.wait_for_run_id("o1", timeout=60)
    assert order.commit_id == "r777"
    assert polls["n"] == 3


def test_wait_for_run_id_raises_when_order_ends_without_one(monkeypatch):
    """A rejected model never gets a run id; waiting out the timeout would
    waste minutes for an outcome already known."""
    monkeypatch.setattr("competition.billing_client.time.sleep", lambda s: None)
    client = BillingClient(base_url="http://billing.invalid/api/v1", api_key="k")
    client.order_status = lambda order_id: Order(
        id=order_id, status="failed", failure_reason="model rejected",
    )

    with pytest.raises(BillingError) as exc:
        client.wait_for_run_id("o1", timeout=60)
    assert exc.value.code == "order_without_run"
    assert "model rejected" in str(exc.value)


def test_wait_for_run_id_times_out(monkeypatch):
    monkeypatch.setattr("competition.billing_client.time.sleep", lambda s: None)
    monotonic = {"t": 0.0}
    monkeypatch.setattr("competition.billing_client.time.monotonic", lambda: monotonic.__setitem__("t", monotonic["t"] + 10) or monotonic["t"])

    client = BillingClient(base_url="http://billing.invalid/api/v1", api_key="k")
    client.order_status = lambda order_id: Order(id=order_id, status="queueing")

    with pytest.raises(BillingError) as exc:
        client.wait_for_run_id("o1", timeout=30)
    assert exc.value.code == "timeout"


# ---------------------------------------------------------------------------
# build_request
# ---------------------------------------------------------------------------

def test_build_request_pins_the_exact_committed_artifact():
    """Validators verify repo, revision and file against the commit, so all
    three must be pinned — a run against "main" would fail verification."""
    request = build_request("user/repo", "a" * 40, "model-Q4.gguf", "mmlu")
    assert request["huggingface_repo"] == "user/repo"
    assert request["huggingface_revision"] == "a" * 40
    assert request["model_files"] == ["model-Q4.gguf"]
    assert request["model_format"] == "gguf"
    assert request["benchmark"] == "mmlu"
