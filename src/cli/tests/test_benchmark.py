"""
`tpn benchmark` — miners run and pay for their own benchmarks.

The money path is what matters here: the miner must see the full bill before
anything is spent, and run ids already paid for must never be lost.
"""
import json

from typer.testing import CliRunner

import cli.commands.benchmark as benchmark_mod
import cli.utils.config as config_mod
from cli.app import app
from competition import leader_config_client
from competition.billing_client import BillingError, Order, Quote

runner = CliRunner()

LEADER_URL = "http://fake-leader"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def stub_leader(monkeypatch, benchmarks=("mmlu", "gsm8k")):
    comp = {
        "id": "comp-a", "name": "comp-a", "start_block": 0, "commit_end_block": 100,
        "scoring_end_block": 200, "emission_distribution": [1.0], "top_n": 1,
        "benchmarks": [{"name": n, "min_score": 0.5} for n in benchmarks],
    }
    leader_config_client._cache.clear()
    leader_config_client._cache_time.clear()
    monkeypatch.setattr(
        leader_config_client.requests, "get",
        lambda url, timeout: FakeResponse({"competitions": [comp]}),
    )


def write_upload_config(monkeypatch, tmp_path, **extra):
    monkeypatch.setattr(config_mod, "tpn_home", lambda: tmp_path / ".tpn")
    cfg = {
        "repository": "user/repo", "file": "model.gguf",
        "huggingface_revision": "a" * 40,
    }
    cfg.update(extra)
    config_mod.save_competition_config("alice", "default", "comp-a", cfg)
    return cfg


def make_quote(**overrides) -> Quote:
    fields = dict(
        charged_cents=650, payable_cents=650, available_credit_cents=5000,
        missing_credit_cents=0, currency="usd", review_required=False,
        confirmation_token="tok",
    )
    fields.update(overrides)
    return Quote(**fields)


class FakeBillingClient:
    """Records the flow so tests can assert on ordering and payloads."""

    def __init__(self, quote=None, submit_error=None, run_ids=None, quote_error=None):
        self._quote = quote or make_quote()
        self._submit_error = submit_error
        self._quote_error = quote_error
        self._run_ids = run_ids or {}
        self.quoted = []
        self.submitted = []

    def quote(self, request, huggingface_token_id=None):
        self.quoted.append(request)
        if self._quote_error:
            raise self._quote_error
        return self._quote

    def submit(self, request, quote=None, huggingface_token_id=None):
        self.submitted.append(request)
        if self._submit_error:
            raise self._submit_error
        return Order(id=f"order-{request['benchmark']}", status="queueing")

    def wait_for_run_id(self, order_id, timeout=600, poll_interval=10):
        benchmark = order_id.replace("order-", "")
        run_id = self._run_ids.get(benchmark, f"r{abs(hash(benchmark)) % 1000}")
        return Order(id=order_id, status="queued", short_id=run_id)


def install_client(monkeypatch, client):
    monkeypatch.setattr(benchmark_mod, "BillingClient", lambda base_url, api_key: client)
    return client


def invoke(*args):
    return runner.invoke(app, ["--leader-url", LEADER_URL, "benchmark",
                               "--wallet", "alice", "--competition", "comp-a", *args])


def saved_runs(tmp_path):
    return config_mod.load_competition_config("alice", "default", "comp-a").get("runs")


# ---------------------------------------------------------------------------
# Guards before any spend
# ---------------------------------------------------------------------------

def test_requires_an_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)
    stub_leader(monkeypatch)
    write_upload_config(monkeypatch, tmp_path)

    result = invoke()
    assert result.exit_code == 1
    assert "No benchmark API key" in result.stdout


def test_requires_upload_data_first(monkeypatch, tmp_path):
    stub_leader(monkeypatch)
    monkeypatch.setattr(config_mod, "tpn_home", lambda: tmp_path / ".tpn")

    result = invoke("--api-key", "bapi_x")
    assert result.exit_code == 1
    assert "Missing upload data" in result.stdout


def test_rejects_benchmarks_not_in_the_competition(monkeypatch, tmp_path):
    """Spending credit on a benchmark the competition never scores is pure
    waste, so it is refused before quoting."""
    stub_leader(monkeypatch)
    write_upload_config(monkeypatch, tmp_path)
    client = install_client(monkeypatch, FakeBillingClient())

    result = invoke("--api-key", "bapi_x", "--only", "not_a_benchmark")
    assert result.exit_code == 1
    assert "Not part of comp-a" in result.stdout
    assert client.quoted == []


def test_refuses_to_submit_when_credit_is_short(monkeypatch, tmp_path):
    """Quoting everything up front means a shortfall is caught before any
    order is placed, rather than halfway through the set."""
    stub_leader(monkeypatch)
    write_upload_config(monkeypatch, tmp_path)
    client = install_client(monkeypatch, FakeBillingClient(
        quote=make_quote(charged_cents=900, payable_cents=900,
                         available_credit_cents=100, missing_credit_cents=800),
    ))

    result = invoke("--api-key", "bapi_x", "--yes")
    assert result.exit_code == 1
    assert "Not enough credit" in result.stdout
    assert client.submitted == []


def test_shows_the_price_and_aborts_when_declined(monkeypatch, tmp_path):
    stub_leader(monkeypatch)
    write_upload_config(monkeypatch, tmp_path)
    client = install_client(monkeypatch, FakeBillingClient())

    result = runner.invoke(
        app,
        ["--leader-url", LEADER_URL, "benchmark", "--wallet", "alice",
         "--competition", "comp-a", "--api-key", "bapi_x"],
        input="n\n",
    )
    assert result.exit_code == 0
    assert "$13.00" in result.stdout  # 2 benchmarks x $6.50
    assert client.submitted == []


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_runs_every_competition_benchmark_and_saves_run_ids(monkeypatch, tmp_path):
    stub_leader(monkeypatch)
    write_upload_config(monkeypatch, tmp_path)
    install_client(monkeypatch, FakeBillingClient(run_ids={"mmlu": "r1", "gsm8k": "r2"}))

    result = invoke("--api-key", "bapi_x", "--yes")
    assert result.exit_code == 0
    assert saved_runs(tmp_path) == [{"b": "mmlu", "r": "r1"}, {"b": "gsm8k", "r": "r2"}]


def test_requests_pin_the_committed_repo_revision_and_file(monkeypatch, tmp_path):
    """These three fields are exactly what the validator verifies, so a run
    produced here must always bind to the commit."""
    stub_leader(monkeypatch, benchmarks=("mmlu",))
    write_upload_config(monkeypatch, tmp_path)
    client = install_client(monkeypatch, FakeBillingClient())

    invoke("--api-key", "bapi_x", "--yes")

    request = client.submitted[0]
    assert request["huggingface_repo"] == "user/repo"
    assert request["huggingface_revision"] == "a" * 40
    assert request["model_files"] == ["model.gguf"]


def test_force_price_skips_quoting(monkeypatch, tmp_path):
    stub_leader(monkeypatch, benchmarks=("mmlu",))
    write_upload_config(monkeypatch, tmp_path)
    client = install_client(monkeypatch, FakeBillingClient())

    result = invoke("--api-key", "bapi_x", "--force-price")
    assert result.exit_code == 0
    assert client.quoted == []
    assert len(client.submitted) == 1


# ---------------------------------------------------------------------------
# Re-running and resuming
# ---------------------------------------------------------------------------

def test_skips_benchmarks_that_already_have_run_ids(monkeypatch, tmp_path):
    """Re-running the command must not silently re-charge for work already
    paid for."""
    stub_leader(monkeypatch)
    write_upload_config(monkeypatch, tmp_path, runs=[
        {"b": "mmlu", "r": "r1"}, {"b": "gsm8k", "r": "r2"},
    ])
    client = install_client(monkeypatch, FakeBillingClient())

    result = invoke("--api-key", "bapi_x", "--yes")
    assert result.exit_code == 0
    assert "already have run ids" in result.stdout
    assert client.submitted == []


def test_only_reruns_the_named_benchmark(monkeypatch, tmp_path):
    """One run id per benchmark exists precisely so a single benchmark can be
    redone without paying for the rest again."""
    stub_leader(monkeypatch)
    write_upload_config(monkeypatch, tmp_path, runs=[
        {"b": "mmlu", "r": "r_old"}, {"b": "gsm8k", "r": "r2"},
    ])
    client = install_client(monkeypatch, FakeBillingClient(run_ids={"mmlu": "r_new"}))

    result = invoke("--api-key", "bapi_x", "--only", "mmlu", "--rerun", "--yes")
    assert result.exit_code == 0
    assert [r["benchmark"] for r in client.submitted] == ["mmlu"]

    runs = {r["b"]: r["r"] for r in saved_runs(tmp_path)}
    assert runs == {"mmlu": "r_new", "gsm8k": "r2"}


def test_fills_in_only_the_missing_benchmark(monkeypatch, tmp_path):
    stub_leader(monkeypatch)
    write_upload_config(monkeypatch, tmp_path, runs=[{"b": "mmlu", "r": "r1"}])
    client = install_client(monkeypatch, FakeBillingClient(run_ids={"gsm8k": "r2"}))

    result = invoke("--api-key", "bapi_x", "--yes")
    assert result.exit_code == 0
    assert [r["benchmark"] for r in client.submitted] == ["gsm8k"]
    assert {r["b"]: r["r"] for r in saved_runs(tmp_path)} == {"mmlu": "r1", "gsm8k": "r2"}


# ---------------------------------------------------------------------------
# Failure partway through
# ---------------------------------------------------------------------------

def test_keeps_run_ids_already_paid_for_when_a_later_one_fails(monkeypatch, tmp_path):
    """The critical money-safety property: a failure on the second benchmark
    must not discard the first, which has already been charged."""
    stub_leader(monkeypatch)
    write_upload_config(monkeypatch, tmp_path)

    class FailsSecond(FakeBillingClient):
        def submit(self, request, quote=None, huggingface_token_id=None):
            if request["benchmark"] == "gsm8k":
                raise BillingError("coordinator rejected the model", code="rejected")
            return super().submit(request, quote)

    install_client(monkeypatch, FailsSecond(run_ids={"mmlu": "r1"}))

    result = invoke("--api-key", "bapi_x", "--yes")
    assert result.exit_code == 1
    assert "Failed: gsm8k" in result.stdout
    assert saved_runs(tmp_path) == [{"b": "mmlu", "r": "r1"}]



