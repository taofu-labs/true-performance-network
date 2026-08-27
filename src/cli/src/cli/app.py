import typer
from rich.console import Console
from common.urls import InvalidBaseUrl, validate_base_url
from cli.commands.register import register
from cli.commands.competitions import competitions
from cli.commands.upload import upload
from cli.commands.commit import commit
from cli.commands.publish import publish
from cli.commands.status import status
from cli.commands.collateral_status import collateral_status
from cli.commands.version import get_version
from typing import Optional
from cli.utils.context import (
    set_context,
    _DEFAULT_NETWORK,
    _DEFAULT_NETUID,
    _DEFAULT_LEADER_URL,
)

app = typer.Typer(
    help="TPN CLI — submit models to the TAO Performance Network.",
    add_completion=False,
)


@app.callback()
def main(
    network: str = typer.Option(
        _DEFAULT_NETWORK,
        "--network",
        help="Bittensor network endpoint (e.g. finney, ws://localhost:9946)",
    ),
    netuid: int = typer.Option(
        _DEFAULT_NETUID,
        "--netuid",
        help="Subnet UID",
    ),
    leader_url: str = typer.Option(
        _DEFAULT_LEADER_URL,
        "--leader-url",
        help="Base URL of the leader validator's API (serves competition configs)",
    ),
    wallet_path: Optional[str] = typer.Option(
        None,
        "--wallet-path",
        help="Path to bittensor wallets directory (default: ~/.bittensor/wallets)",
    ),
    block_time: float = typer.Option(
        12.0,
        "--block-time",
        help="Seconds per block (default 12.0 for mainnet; use 0.25 for fast-runtime localnet)",
    ),
):
    try:
        leader_url = validate_base_url(leader_url, setting_name="--leader-url")
    except InvalidBaseUrl as e:
        Console().print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    set_context(network=network, netuid=netuid, leader_url=leader_url, wallet_path=wallet_path, block_time=block_time)


app.command("register")(register)
app.command("competitions")(competitions)
app.command("upload")(upload)
app.command("commit")(commit)
app.command("publish")(publish)
app.command("status")(status)
app.command("collateral-status")(collateral_status)
app.command("version")(get_version)

if __name__ == "__main__":
    app()
