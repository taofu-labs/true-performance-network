import os

import os as _os
from common.settings import BITTENSOR

WEIGHT_SUBMIT_INTERVAL: int = int(_os.getenv("WEIGHT_SUBMIT_INTERVAL", 10 if not BITTENSOR else 60 * 21))
ORCHESTRATOR_HEALTH_CHECK_INTERVAL: int = 60
VALIDATOR_LOOP_INTERVAL: int = int(_os.getenv("VALIDATOR_LOOP_INTERVAL", 10 if not BITTENSOR else 60))

# Health settings
LAUNCH_HEALTH = os.getenv("LAUNCH_HEALTH") == "True"
VALIDATOR_HEALTH_HOST = os.getenv("VALIDATOR_HEALTH_HOST", "0.0.0.0")
VALIDATOR_HEALTH_PORT = int(os.getenv("VALIDATOR_HEALTH_PORT", 9100))
VALIDATOR_HEALTH_ENDPOINT = os.getenv("VALIDATOR_HEALTH_ENDPOINT", "/health")

WALLET_COLDKEY = os.getenv("WALLET_COLDKEY", "test")
WALLET_HOTKEY = os.getenv("WALLET_HOTKEY", "m1")
WALLET_PATH = os.getenv("WALLET_PATH", None)  # None = use bittensor default (~/.bittensor/wallets)

REQUEST_RETRY_COUNT = int(os.getenv("REQUEST_RETRY_COUNT", 3))
CLIENT_REQUEST_TIMEOUT = int(os.getenv("CLIENT_REQUEST_TIMEOUT", 30))

# Benchmark submit and poll have different coordinator capacity profiles —
# submitting many jobs at once can hit rate/queue limits even though polling
# them in parallel afterward is fine. Caps concurrent *submits* only; polling
# stays unbounded/parallel once a run_id is accepted.
BENCHMARK_SUBMIT_CONCURRENCY: int = int(os.getenv("BENCHMARK_SUBMIT_CONCURRENCY", 1))

SCORING_QUEUE_CONCURRENCY: int = int(os.getenv("SCORING_QUEUE_CONCURRENCY", 5))

# Minimum alpha a hotkey must have locked as miner collateral to be scored
# this run. Not a ban — a miner below threshold is simply skipped for the
# current scoring pass; topping up collateral before the next pass restores
# scoring. 0 = disabled.
COLLATERAL_MIN_THRESHOLD: float = float(os.getenv("COLLATERAL_MIN_THRESHOLD", 0))

# Leader API (mode=leader only) — serves read-only scoring state to followers/dashboards
LEADER_API_HOST = os.getenv("LEADER_API_HOST", "0.0.0.0")
LEADER_API_PORT = int(os.getenv("LEADER_API_PORT", 9200))

# Follower client (mode=follower only) — URL of the leader validator's read API
LEADER_VALIDATOR_URL = os.getenv("LEADER_VALIDATOR_URL", "")
FOLLOWER_POLL_INTERVAL = int(os.getenv("FOLLOWER_POLL_INTERVAL", 60))

# Admin API (mode=leader only) — bearer token gating POST /v1/competitions.
# Unset = write endpoint refuses all requests (503), not open (401 only on bad/missing token).
ADMIN_API_KEY: str | None = os.getenv("ADMIN_API_KEY") or None
