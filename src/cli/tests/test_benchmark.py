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


def stub_leader(monkeypatch, benchmarks=("mmlu", "gsm8k"), **extra):
    comp = {
        "id": "comp-a", "name": "comp-a", "start_block": 0, "commit_end_block": 100,
        "scoring_end_block": 200, "emission_distribution": [1.0], "top_n": 1,
        "benchmarks": [{"name": n, "min_score": 0.5} for n in benchmarks],
    }
    comp.update(extra)
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


def run_name(request):
    """The run a request is for: a benchmark by name, or the KL comparison."""
    return request["benchmark"] if "benchmark" in request else "kl"


class FakeBillingClient:
    """Records the flow so tests can assert on ordering and payloads."""

    def __init__(self, quote=None, submit_error=None, run_ids=None, quote_error=None):
        self._quote = quote or make_quote()
        self._submit_error = submit_error
        self._quote_error = quote_error
        self._run_ids = run_ids or {}
        self.quoted = []
        self.quote_paths = []
        self.submit_paths = []
        self.submitted = []

    def quote(self, request, huggingface_token_id=None, path="/benchmarks/quote"):
        self.quoted.append(request)
        self.quote_paths.append(path)
        if self._quote_error:
            raise self._quote_error
        return self._quote

    def submit(self, request, quote=None, huggingface_token_id=None, path="/benchmarks"):
        self.submitted.append(request)
        self.submit_paths.append(path)
        if self._submit_error:
            raise self._submit_error
        return Order(id=f"order-{run_name(request)}", status="queueing")

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
        def submit(self, request, quote=None, huggingface_token_id=None, path="/benchmarks"):
            if request.get("benchmark") == "gsm8k":
                raise BillingError("coordinator rejected the model", code="rejected")
            return super().submit(request, quote, path=path)

    install_client(monkeypatch, FailsSecond(run_ids={"mmlu": "r1"}))

    result = invoke("--api-key", "bapi_x", "--yes")
    assert result.exit_code == 1
    assert "Failed: gsm8k" in result.stdout
    assert saved_runs(tmp_path) == [{"b": "mmlu", "r": "r1"}]





# ── KL check ──────────────────────────────────────────────────────────────────
# A competition with kl_corpus_id requires a comparison run alongside its
# benchmarks. It is billed through /comparisons, not /benchmarks.

KL_FIELDS = {"kl_corpus_id": "corpus-000002", "model_repo": "Qwen/Qwen2.5-0.5B-Instruct"}


def test_runs_the_kl_check_by_default(monkeypatch, tmp_path):
    stub_leader(monkeypatch, **KL_FIELDS)
    write_upload_config(monkeypatch, tmp_path)
    client = FakeBillingClient(run_ids={"mmlu": "r1", "gsm8k": "r2", "kl": "r3"})
    install_client(monkeypatch, client)

    result = invoke("--api-key", "bapi_x", "--yes")
    assert result.exit_code == 0, result.stdout
    saved = {r["b"]: r["r"] for r in saved_runs(tmp_path)}
    assert saved == {"mmlu": "r1", "gsm8k": "r2", "kl": "r3"}


def test_kl_check_is_billed_through_the_comparison_endpoints(monkeypatch, tmp_path):
    stub_leader(monkeypatch, **KL_FIELDS)
    write_upload_config(monkeypatch, tmp_path)
    client = FakeBillingClient()
    install_client(monkeypatch, client)

    invoke("--api-key", "bapi_x", "--yes")
    assert client.quote_paths.count("/comparisons/quote") == 1
    assert client.submit_paths.count("/comparisons") == 1
    assert client.quote_paths.count("/benchmarks/quote") == 2


def test_kl_request_pins_the_base_the_corpus_and_the_committed_file(monkeypatch, tmp_path):
    stub_leader(monkeypatch, **KL_FIELDS)
    write_upload_config(monkeypatch, tmp_path, file="model-Q4_K_M.gguf")
    client = FakeBillingClient()
    install_client(monkeypatch, client)

    invoke("--api-key", "bapi_x", "--yes")
    kl = next(r for r in client.submitted if "benchmark" not in r)
    assert kl["protocol"] == "fidelity-v1"
    # A gguf reference has no base-side pairing, so the base could not be checked.
    assert kl["reference"] == "transformers"
    assert kl["base"]["huggingface_repo"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert kl["dataset"] == {"kind": "platform", "corpus_id": "corpus-000002"}
    assert kl["candidate"]["huggingface_repo"] == "user/repo"
    assert kl["candidate"]["huggingface_revision"] == "a" * 40
    assert kl["candidate"]["model_files"] == ["model-Q4_K_M.gguf"]
    assert kl["candidate"]["quantization"] == "Q4_K_M"


def test_kl_price_is_in_the_bill_the_miner_confirms(monkeypatch, tmp_path):
    stub_leader(monkeypatch, **KL_FIELDS)
    write_upload_config(monkeypatch, tmp_path)
    install_client(monkeypatch, FakeBillingClient())

    result = runner.invoke(app, ["--leader-url", LEADER_URL, "benchmark",
                                 "--wallet", "alice", "--competition", "comp-a",
                                 "--api-key", "bapi_x"], input="n\n")
    assert result.exit_code == 0
    assert "kl" in result.stdout and "KL check" in result.stdout
    # Three runs at $6.50; nothing is spent when the miner declines.
    assert "$19.50" in result.stdout


def test_only_kl_runs_the_comparison_alone(monkeypatch, tmp_path):
    stub_leader(monkeypatch, **KL_FIELDS)
    write_upload_config(monkeypatch, tmp_path)
    client = FakeBillingClient(run_ids={"kl": "r9"})
    install_client(monkeypatch, client)

    result = invoke("--api-key", "bapi_x", "--only", "kl", "--yes")
    assert result.exit_code == 0, result.stdout
    assert client.quote_paths == ["/comparisons/quote"]
    assert {r["b"]: r["r"] for r in saved_runs(tmp_path)} == {"kl": "r9"}


def test_no_kl_check_when_the_competition_has_no_corpus(monkeypatch, tmp_path):
    stub_leader(monkeypatch)
    write_upload_config(monkeypatch, tmp_path)
    client = FakeBillingClient()
    install_client(monkeypatch, client)

    invoke("--api-key", "bapi_x", "--yes")
    assert "/comparisons/quote" not in client.quote_paths
    assert "kl" not in {r["b"] for r in saved_runs(tmp_path)}


def test_refuses_a_kl_competition_that_pins_no_base(monkeypatch, tmp_path):
    """CompetitionSpec now refuses kl_corpus_id without model_repo, so such a
    competition never parses and the CLI cannot resolve it at all. The miner
    gets a clear "not found" rather than paying for an unverifiable KL run.

    The guard inside `benchmark` stays as defence in depth — it is what catches
    a spec that reaches the CLI some other way.
    """
    stub_leader(monkeypatch, kl_corpus_id="corpus-000002")
    write_upload_config(monkeypatch, tmp_path)
    install_client(monkeypatch, FakeBillingClient())

    result = invoke("--api-key", "bapi_x", "--yes")
    assert result.exit_code == 1
    assert "comp-a" in result.stdout


def test_runs_a_kl_competition_that_pins_a_base(monkeypatch, tmp_path):
    """The same competition, correctly configured, reaches the KL run."""
    stub_leader(monkeypatch, kl_corpus_id="corpus-000002", model_repo="Qwen/Qwen2.5-0.5B-Instruct")
    write_upload_config(monkeypatch, tmp_path)
    client = FakeBillingClient()
    install_client(monkeypatch, client)

    result = invoke("--api-key", "bapi_x", "--yes")
    assert result.exit_code == 0, result.stdout
    assert "kl" in result.stdout
