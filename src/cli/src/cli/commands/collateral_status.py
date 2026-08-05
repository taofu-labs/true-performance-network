import typer
from rich.console import Console
from rich.panel import Panel
from cli.utils.context import get as get_ctx

console = Console()


def collateral_status(
    coldkey: str = typer.Option(..., "--wallet", "-w", help="Bittensor wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey", help="Hotkey name"),
):
    """Show your current miner collateral status."""
    ctx = get_ctx()
    from common.chain import get_subtensor, get_wallet

    wallet = get_wallet(coldkey=coldkey, hotkey=hotkey_name, wallet_path=ctx.wallet_path)
    subtensor = get_subtensor(ctx.network)

    data = subtensor.read(
        "miner_collateral",
        netuid=ctx.netuid,
        hotkey_ss58=wallet.hotkey.ss58_address,
    )

    if data is None:
        console.print(Panel(
            "No collateral position found.",
            title="Collateral Status",
            border_style="yellow",
        ))
        return

    lines = [
        f"[dim]Locked:[/dim]            {data['locked_alpha']}",
        f"[dim]Floor (min):[/dim]       {data['min_locked_alpha']}",
        f"[dim]Lifetime earned:[/dim]   {data['earned_alpha']}",
        f"[dim]Drain ratio:[/dim]       {data['drain_ratio']:.4f}",
        f"[dim]Headroom:[/dim]          {data['headroom_alpha']}",
        f"[dim]Shortfall:[/dim]         {data['shortfall_alpha']}",
    ]
    console.print(Panel("\n".join(lines), title="Collateral Status", border_style="cyan"))
