"""Global CLI options resolved once at app entry, read by all commands."""
from dataclasses import dataclass
from typing import Optional

_DEFAULT_NETWORK = "finney"
_DEFAULT_NETUID = 65
_DEFAULT_COMPETITION_URL = (
    "https://raw.githubusercontent.com/taofu-labs/tao-performance-network/main/competitions/index.json"
)


@dataclass
class CLIContext:
    network: str = _DEFAULT_NETWORK
    netuid: int = _DEFAULT_NETUID
    competition_url: str = _DEFAULT_COMPETITION_URL
    wallet_path: Optional[str] = None  # None = bittensor default (~/.bittensor/wallets)
    block_time: float = 12.0  # seconds per block; use 0.25 for fast-runtime localnet


# Module-level singleton set by app callback, read by commands.
_ctx = CLIContext()


def get() -> CLIContext:
    return _ctx


def set_context(network: str, netuid: int, competition_url: str, wallet_path: Optional[str], block_time: float = 12.0) -> None:
    _ctx.network = network
    _ctx.netuid = netuid
    _ctx.competition_url = competition_url
    _ctx.wallet_path = wallet_path
    _ctx.block_time = block_time
