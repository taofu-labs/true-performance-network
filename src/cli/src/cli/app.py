import typer
from cli.commands.register import register
from cli.commands.competitions import competitions
from cli.commands.upload import upload
from cli.commands.commit import commit
from cli.commands.publish import publish
from cli.commands.status import status
from cli.commands.version import get_version
from typing import Optional
from cli.utils.context import (
    set_context,
    _DEFAULT_NETWORK,
    _DEFAULT_NETUID,
    _DEFAULT_COMPETITION_URL,
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
    competition_url: str = typer.Option(
        _DEFAULT_COMPETITION_URL,
        "--competition-url",
        help="URL or path to competitions index.json (https://... or /abs/path/index.json)",
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
    set_context(network=network, netuid=netuid, competition_url=competition_url, wallet_path=wallet_path, block_time=block_time)


app.command("register")(register)
app.command("competitions")(competitions)
app.command("upload")(upload)
app.command("commit")(commit)
app.command("publish")(publish)
app.command("status")(status)
app.command("version")(get_version)

if __name__ == "__main__":
    app()
