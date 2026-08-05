"""
Wrapper around btcli subnet register.
Guides the miner through registration on the TPN subnet, then optionally
locks miner collateral and sets a self-maintaining collateral floor.
"""
import subprocess
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from cli.utils.config import init_identity_dir
from cli.utils.context import get as get_ctx

console = Console()


def _report_collateral_result(label: str, result) -> None:
    if result.success:
        console.print(f"[green]✓ {label}[/green]")
    else:
        console.print(f"[red]✗ {label} failed: {result.message}[/red]")


def register(
    coldkey: str = typer.Option(..., prompt=True, help="Bittensor wallet name"),
    hotkey: str = typer.Option("default", prompt=True, help="Hotkey name"),
    collateral: bool = typer.Option(
        True, "--collateral/--no-collateral",
        help="Lock additional alpha as miner collateral after registering",
    ),
    collateral_amount: Optional[float] = typer.Option(
        None, "--collateral-amount",
        help="Alpha to lock as collateral (prompted if --collateral and not set)",
    ),
    floor: bool = typer.Option(
        True, "--floor/--no-floor",
        help="Set a self-maintaining collateral floor after registering",
    ),
    floor_amount: Optional[float] = typer.Option(
        None, "--floor-amount",
        help="Collateral floor in alpha (prompted if --floor and not set)",
    ),
):
    """Register your hotkey on the TAO Performance Network subnet."""
    ctx = get_ctx()

    console.print(Panel(
        f"Registering on subnet [cyan]{ctx.netuid}[/cyan] ({ctx.network})\n"
        f"Wallet: [cyan]{coldkey}[/cyan] / Hotkey: [cyan]{hotkey}[/cyan]\n\n"
        f"[yellow]This will cost TAO. Confirm in the next prompt.[/yellow]",
        title="Subnet Registration",
        border_style="yellow",
    ))

    if not typer.confirm("Proceed?"):
        raise typer.Exit(0)

    for val, name in [(coldkey, "coldkey"), (hotkey, "hotkey")]:
        if val.startswith("-"):
            console.print(f"[red]Invalid {name}: must not start with '-'[/red]")
            raise typer.Exit(1)

    cmd = [
        "btcli", "subnet", "register",
        "--wallet", coldkey,
        "--wallet-hotkey", hotkey,
        "--netuid", str(ctx.netuid),
        "--network", ctx.network,
        "--yes",
    ]
    if ctx.wallet_path:
        cmd += ["--wallet-path", ctx.wallet_path]

    console.print(f"[blue]Running: {' '.join(cmd)}[/blue]")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        config_dir = init_identity_dir(coldkey, hotkey)
        console.print(Panel(
            f"[green]✓ Registration successful[/green]\n\n"
            f"Config directory: [dim]{config_dir}[/dim]",
            title="Registered",
            border_style="green",
        ))

        if collateral or floor:
            from common.chain import get_subtensor, get_wallet
            from bittensor.intents import AddCollateral, SetMinCollateral

            subtensor = get_subtensor(ctx.network)
            wallet = get_wallet(coldkey=coldkey, hotkey=hotkey, wallet_path=ctx.wallet_path)

            if collateral:
                amount = collateral_amount if collateral_amount is not None else typer.prompt(
                    "Alpha to lock as collateral (0 to skip)", type=float
                )
                if amount > 0:
                    collateral_result = subtensor.submit_shielded(
                        AddCollateral(netuid=ctx.netuid, amount_alpha=amount),
                        wallet,
                    )
                    _report_collateral_result("Collateral locked", collateral_result)

            if floor:
                amount = floor_amount if floor_amount is not None else typer.prompt(
                    "Collateral floor in alpha (0 to skip)", type=float
                )
                if amount > 0:
                    floor_result = subtensor.execute(
                        SetMinCollateral(netuid=ctx.netuid, min_alpha=amount),
                        wallet,
                    )
                    _report_collateral_result("Collateral floor set", floor_result)
    else:
        console.print("[red]✗ Registration failed. Check btcli output above.[/red]")
        raise typer.Exit(1)
