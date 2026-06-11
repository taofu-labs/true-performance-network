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
