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

# Competition config index — points directly at index.json
COMPETITION_INDEX_URL = os.getenv(
    "COMPETITION_INDEX_URL",
    "https://raw.githubusercontent.com/taofu-labs/tao-performance-network/main/competitions/index.json",
)
COMPETITION_REFRESH_INTERVAL = int(os.getenv("COMPETITION_REFRESH_INTERVAL", 600))

# HuggingFace — optional, only set on the validator that publishes the registry
# If unset, registry publication is skipped silently
HF_TOKEN: str | None = os.getenv("HF_TOKEN") or None
HF_ORG: str | None = os.getenv("HF_ORG") or None

# Minimum locked alpha a coldkey must have on the subnet hotkey to be eligible for scoring.
# Compared against LockState.locked_mass (Balance in subnet alpha units).
MIN_ALPHA_LOCK: float = float(os.getenv("MIN_ALPHA_LOCK", 100.0))
