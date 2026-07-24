import os
from dotenv import load_dotenv
from loguru import logger

__SPEC_VERSION__ = 100000

TPN_DOTENV_PATH = os.getenv("TPN_DOTENV_PATH")
if TPN_DOTENV_PATH:
    if not load_dotenv(dotenv_path=TPN_DOTENV_PATH):
        logger.warning(f"No .env file found at {TPN_DOTENV_PATH}")
else:
    load_dotenv()  # silently try default .env; no warning if absent

# Bittensor
BITTENSOR = os.getenv("BITTENSOR", "True") == "True"
NETUID = int(os.getenv("NETUID", 0))
NETWORK = os.getenv("NETWORK", "finney")

# Validator role: "leader" runs chain scan/precheck/benchmark/score and serves the read API.
# "follower" reads scoring results from a leader validator and sets weights from them.
VALIDATOR_MODE: str = os.getenv("VALIDATOR_MODE", "leader")

# HuggingFace — optional, only set on the validator that publishes the registry
# If unset, registry publication is skipped silently
HF_TOKEN: str | None = os.getenv("HF_TOKEN") or None
HF_ORG: str | None = os.getenv("HF_ORG") or None

# Benchmark coordinator
BENCHMARK_BACKEND: str = os.getenv("BENCHMARK_BACKEND", "mock")  # "mock" | "http"
COORDINATOR_BASE_URL: str = os.getenv("COORDINATOR_BASE_URL", "https://bench.trueperformancenetwork.com/api/coordinator")
COORDINATOR_API_KEY: str | None = os.getenv("COORDINATOR_API_KEY") or None

# Precheck container (provenance + RAM check)
PRECHECK_IMAGE: str = os.getenv("PRECHECK_IMAGE", "tpn-precheck")
PRECHECK_HOST_PORT: int = int(os.getenv("PRECHECK_HOST_PORT", "8081"))
RAM_CHECK_LYING_TOLERANCE: float = float(os.getenv("RAM_CHECK_LYING_TOLERANCE", "0.01"))

# Max wall-clock seconds to poll a single benchmark run before giving up (skip, not fail-fast)
BENCHMARK_POLL_TIMEOUT_SECONDS: int = int(os.getenv("BENCHMARK_POLL_TIMEOUT_SECONDS", "5400"))
