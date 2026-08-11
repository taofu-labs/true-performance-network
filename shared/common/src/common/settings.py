import os
import sys
from dotenv import load_dotenv
from loguru import logger

__SPEC_VERSION__ = 100000

TPN_DOTENV_PATH = os.getenv("TPN_DOTENV_PATH")
if TPN_DOTENV_PATH:
    if not load_dotenv(dotenv_path=TPN_DOTENV_PATH):
        logger.warning(f"No .env file found at {TPN_DOTENV_PATH}")
else:
    load_dotenv()  # silently try default .env; no warning if absent

# Logging — DEBUG is dev-only (set via .env.localnet); prod stays at INFO.
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

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
PRECHECK_IMAGE: str = os.getenv("PRECHECK_IMAGE", "ghcr.io/taofu-labs/tpn-precheck")
PRECHECK_HOST_PORT: int = int(os.getenv("PRECHECK_HOST_PORT", "8081"))
RAM_CHECK_LYING_TOLERANCE: float = float(os.getenv("RAM_CHECK_LYING_TOLERANCE", "0.01"))
SCORE_LYING_TOLERANCE: float = float(os.getenv("SCORE_LYING_TOLERANCE", "0.01"))

# Stale-progress windows: a benchmark run is considered stuck only when the
# coordinator's progress.last_log_at stops advancing for longer than the
# window for its current phase bucket — not by total elapsed wall-clock time.
# Startup (provisioning/model download/preflight) gets a shorter window than
# execution (benchmarking/collecting_results), where lm_eval heartbeats are
# coarser and multi-hour runs (MMLU, HellaSwag) are expected and healthy.
BENCHMARK_STARTUP_STALE_SECONDS: int = int(os.getenv("BENCHMARK_STARTUP_STALE_SECONDS", "900"))
BENCHMARK_EXECUTION_STALE_SECONDS: int = int(os.getenv("BENCHMARK_EXECUTION_STALE_SECONDS", "1800"))


def configure_logging() -> None:
    """Point loguru's default stderr handler at LOG_LEVEL.

    Call once from a process entrypoint (validator/CLI main), not on import —
    logging setup is a process-level side effect, not a settings default.
    """
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL)
