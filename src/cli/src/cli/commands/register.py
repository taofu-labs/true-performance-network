"""
Wrapper around btcli subnet register.
Guides the miner through registration on the TPN subnet.
"""
import subprocess
import typer
from rich.console import Console
from rich.panel import Panel
from cli.utils.config import init_identity_dir
from cli.utils.context import get as get_ctx

console = Console()


def register(
    coldkey: str = typer.Option(..., prompt=True, help="Bittensor wallet name"),
    hotkey: str = typer.Option("default", prompt=True, help="Hotkey name"),
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
        "--wallet.name", coldkey,
        "--wallet.hotkey", hotkey,
        "--netuid", str(ctx.netuid),
        "--subtensor.network", ctx.network,
    ]
    if ctx.wallet_path:
        cmd += ["--wallet.path", ctx.wallet_path]

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
    else:
        console.print("[red]✗ Registration failed. Check btcli output above.[/red]")
        raise typer.Exit(1)
