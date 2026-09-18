"""
Miner-side client for the benchmark service's billing API.

Miners run their own benchmarks and commit the resulting coordinator run ids on
chain. This client drives the quote -> submit -> poll flow with a billing API
key (`bapi_...`), issued from the benchmark service's Tokens page.

Billing keys are scoped to `/api/v1` only: they cannot call the coordinator
directly, and they cannot buy credits. Top up through the web GUI first.

    client = BillingClient(base_url, api_key)
    quote = client.quote(request)             # price + confirmation token
    order = client.submit(request, quote)     # spends credit, returns order id
    status = client.order_status(order_id)    # run_id lands here once queued
"""
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from loguru import logger

from common.urls import validate_base_url

DEFAULT_BILLING_URL = os.getenv("BENCHMARK_BILLING_URL", "https://benchmarks.trueperformancenetwork.com/api/v1")

# Order statuses that will never progress further.
_TERMINAL_ORDER_STATUSES = {"completed", "failed", "cancelled", "refunded"}


class BillingError(Exception):
    """A billing API call failed. `code` is the service's machine-readable error code."""

    def __init__(self, message: str, code: str = "", status_code: int = 0, detail: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.detail = detail or {}


@dataclass
class Quote:
    charged_cents: int
    payable_cents: int
    available_credit_cents: int
    missing_credit_cents: int
    currency: str
    review_required: bool
    confirmation_token: Optional[str]
    estimated_runtime_seconds: Optional[float] = None
    reuse_eligible: bool = False


@dataclass
class Order:
    id: str
    status: str
    charged_cents: int = 0
    requires_admin_review: bool = False
    # Present once the coordinator has accepted the work. This is the value the
    # miner commits on chain — not the order id.
    run_id: Optional[str] = None
    short_id: Optional[str] = None
    run_status: Optional[str] = None
    failure_reason: Optional[str] = None

    @property
    def commit_id(self) -> Optional[str]:
        """Run id to put on chain. Prefers the short id — it costs less payload."""
        return self.short_id or self.run_id

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_ORDER_STATUSES


class BillingClient:
    def __init__(self, base_url: str = "", api_key: str = "", timeout: int = 60):
        base = base_url or DEFAULT_BILLING_URL
        self._base = validate_base_url(base, setting_name="BENCHMARK_BILLING_URL")
        if not api_key:
            raise BillingError("A benchmark API key is required (BENCHMARK_API_KEY).", code="missing_api_key")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self._timeout = timeout

    def _call(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self._base}{path}"
        try:
            resp = self._session.request(method, url, timeout=self._timeout, **kwargs)
        except requests.RequestException as e:
            raise BillingError(f"cannot reach benchmark service: {e}", code="unreachable") from e

        if resp.status_code >= 400:
            try:
                error = resp.json().get("error") or {}
            except ValueError:
                error = {}
            raise BillingError(
                error.get("message") or f"HTTP {resp.status_code}",
                code=error.get("code", ""),
                status_code=resp.status_code,
                detail=error.get("details") or {},
            )
        try:
            return resp.json()
        except ValueError as e:
            raise BillingError("benchmark service returned a non-JSON response", code="bad_response") from e

    # ── Capabilities ──────────────────────────────────────────────────────

    def capabilities(self) -> dict:
        """Supported benchmarks, engines and model formats."""
        return self._call("GET", "/capabilities")

    def account(self) -> dict:
        """Current session/account, including available credit."""
        return self._call("GET", "/me")

    # ── Quote / submit / poll ─────────────────────────────────────────────

    def quote(self, request: dict, huggingface_token_id: Optional[str] = None) -> Quote:
        body: dict = {"request": request}
        if huggingface_token_id:
            body["huggingface_token_id"] = huggingface_token_id
        data = self._call("POST", "/benchmarks/quote", json=body)
        price = data.get("billing_price") or {}
        coordinator_quote = data.get("quote") or {}
        return Quote(
            charged_cents=int(price.get("charged_cents") or 0),
            payable_cents=int(price.get("payable_cents") or 0),
            available_credit_cents=int(price.get("available_credit_cents") or 0),
            missing_credit_cents=int(price.get("missing_credit_cents") or 0),
            currency=price.get("currency") or "usd",
            review_required=bool(price.get("review_required")),
            confirmation_token=data.get("quote_confirmation_token"),
            estimated_runtime_seconds=coordinator_quote.get("estimated_runtime_seconds"),
            reuse_eligible=bool((data.get("reuse") or {}).get("eligible")),
        )

    def submit(
        self,
        request: dict,
        quote: Optional[Quote] = None,
        huggingface_token_id: Optional[str] = None,
    ) -> Order:
        """
        Place a benchmark order, spending prepaid credit.

        Passing the quote echoes its signed confirmation token and asserts the
        price the miner was shown. Without one, `force` accepts whatever the
        service currently charges — API keys only, and still rate limited.
        """
        body: dict = {"request": request}
        if huggingface_token_id:
            body["huggingface_token_id"] = huggingface_token_id
        if quote and quote.confirmation_token:
            body["quote_confirmation_token"] = quote.confirmation_token
            body["expected_charged_cents"] = quote.charged_cents
        else:
            body["force"] = True

        data = self._call("POST", "/benchmarks", json=body)
        order = data.get("order") or {}
        return Order(
            id=order.get("id", ""),
            status=order.get("status", ""),
            charged_cents=int(order.get("charged_cents") or 0),
            requires_admin_review=bool(order.get("requires_admin_review")),
        )

    def order_status(self, order_id: str) -> Order:
        data = self._call("GET", f"/orders/{order_id}/status")
        order = data.get("order") or {}
        run = data.get("coordinator_status") or {}
        return Order(
            id=order.get("id", order_id),
            status=order.get("status", ""),
            charged_cents=int(order.get("charged_cents") or 0),
            requires_admin_review=bool(order.get("requires_admin_review")),
            run_id=run.get("run_id"),
            short_id=run.get("short_id"),
            run_status=run.get("status"),
            failure_reason=run.get("failure_reason") or order.get("failure_classification"),
        )

    def wait_for_run_id(self, order_id: str, timeout: int = 600, poll_interval: int = 10) -> Order:
        """
        Poll an order until the coordinator assigns it a run id.

        The run id appears as soon as the work is queued — long before the
        benchmark finishes — so this returns quickly and does not wait for a
        score. An order that reaches a terminal state without a run id failed.
        """
        deadline = time.monotonic() + timeout
        # The coordinator usually assigns a run id within a second or two of
        # accepting the order, so back off gently rather than paying the full
        # interval on the common case.
        delay = 1.0
        while True:
            order = self.order_status(order_id)
            if order.commit_id:
                return order
            if order.is_terminal:
                raise BillingError(
                    f"order {order_id} ended as '{order.status}' without a run id"
                    + (f": {order.failure_reason}" if order.failure_reason else ""),
                    code="order_without_run",
                )
            if time.monotonic() >= deadline:
                raise BillingError(
                    f"order {order_id} has no run id after {timeout}s (status={order.status})",
                    code="timeout",
                )
            logger.debug(f"order {order_id} status={order.status} — waiting for run id")
            time.sleep(delay)
            delay = min(delay * 2, poll_interval)


def build_request(
    repository: str,
    revision: str,
    file: str,
    benchmark: str,
    benchmark_engine: str = "lm_eval",
) -> dict:
    """
    Build a benchmark request pinned to exactly the artifact a miner commits.

    Validators verify the run's repo, revision and file against the on-chain
    commit, so all three must be the committed ones — a run against `main`, or
    against a different .gguf in the same repo, scores 0.0.
    """
    return {
        "huggingface_repo": repository,
        "huggingface_revision": revision,
        "model_format": "gguf",
        "model_files": [file],
        "benchmark_engine": benchmark_engine,
        "benchmark": benchmark,
    }
