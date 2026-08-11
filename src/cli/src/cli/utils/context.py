"""Global CLI options resolved once at app entry, read by all commands."""
from dataclasses import dataclass
from typing import Optional

_DEFAULT_NETWORK = "finney"
_DEFAULT_NETUID = 65
_DEFAULT_LEADER_URL = "https://leader.tpn.internal"


@dataclass
class CLIContext:
    network: str = _DEFAULT_NETWORK
    netuid: int = _DEFAULT_NETUID
    leader_url: str = _DEFAULT_LEADER_URL
    wallet_path: Optional[str] = None  # None = bittensor default (~/.bittensor/wallets)
    block_time: float = 12.0  # seconds per block; use 0.25 for fast-runtime localnet


# Module-level singleton set by app callback, read by commands.
_ctx = CLIContext()


def get() -> CLIContext:
    return _ctx


def set_context(network: str, netuid: int, leader_url: str, wallet_path: Optional[str], block_time: float = 12.0) -> None:
    _ctx.network = network
    _ctx.netuid = netuid
    _ctx.leader_url = leader_url
    _ctx.wallet_path = wallet_path
    _ctx.block_time = block_time


def current_block_safe(ctx: CLIContext) -> int:
    """Best-effort current block; returns 0 if the chain is unreachable."""
    try:
        from common.chain import current_block, get_subtensor
        return current_block(get_subtensor(ctx.network))
    except Exception:
        return 0


def resolve_competition_or_exit(ctx: CLIContext, competition_id: str):
    """Fetch a competition spec by ID, or print an error and exit(1) if it doesn't exist."""
    import typer
    from rich.console import Console
    from competition.leader_config_client import get_competition_by_id

    spec = get_competition_by_id(base_url=ctx.leader_url, competition_id=competition_id)
    if spec is None:
        Console().print(f"[red]Competition '{competition_id}' not found.[/red]")
        raise typer.Exit(1)
    return spec
